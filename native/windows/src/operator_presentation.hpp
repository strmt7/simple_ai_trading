#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace simple_ai_trading::presentation {

// These describe observations, never financial readiness or order authority.
enum class Tone { unknown, observed, caution, critical, simulation };

struct Status {
    std::wstring label = L"Unavailable";
    Tone tone = Tone::unknown;
};

inline Status runtime_display(std::wstring_view venue, Status observation, bool simulated) {
    if (simulated) return {std::wstring(venue) + L" simulated", Tone::simulation};
    return observation;
}

inline bool unsigned_count(std::wstring_view text, std::uint64_t& value) {
    if (text.empty()) return false;
    value = 0;
    for (wchar_t c : text) {
        if (c < L'0' || c > L'9') return false;
        const auto digit = static_cast<std::uint64_t>(c - L'0');
        if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) return false;
        value = value * 10 + digit;
    }
    return true;
}

inline Status api_budget(std::wstring_view line) {
    if (line.starts_with(L"dry-run: ")) return {L"Dry-run", Tone::simulation};
    constexpr std::wstring_view prefix = L"API budget: ";
    if (!line.starts_with(prefix)) return {};
    const auto body = line.substr(prefix.size());
    const auto state = body.substr(0, body.find(L' '));
    if (state == L"blocked") return {L"Blocked", Tone::critical};
    if (state != L"ok" && state != L"warn" && state != L"critical") return {};

    constexpr std::wstring_view marker = L" remaining=";
    const auto at = body.find(marker);
    if (at == std::wstring_view::npos) {
        if (state == L"critical") return {L"Critical", Tone::critical};
        if (state == L"warn") return {L"Low reserve", Tone::caution};
        return {};
    }
    const auto rest = body.substr(at + marker.size());
    const auto fraction = rest.substr(0, rest.find(L' '));
    const auto slash = fraction.find(L'/');
    std::uint64_t remaining = 0, limit = 0;
    if (slash == std::wstring_view::npos ||
        !unsigned_count(fraction.substr(0, slash), remaining) ||
        !unsigned_count(fraction.substr(slash + 1), limit) || limit == 0 || remaining > limit) {
        if (state == L"critical") return {L"Critical", Tone::critical};
        if (state == L"warn") return {L"Low reserve", Tone::caution};
        return {};
    }
    // A zero sample must remain conspicuous even if its upstream status conflicts.
    const auto tone = state == L"critical" || remaining == 0 ? Tone::critical
                      : state == L"warn" ? Tone::caution : Tone::observed;
    const std::wstring label = tone == Tone::critical ? L"Critical: "
                               : tone == Tone::caution ? L"Low: " : L"";
    return {label + std::wstring(fraction) + L" remaining", tone};
}

inline Status compute(std::wstring_view backend) {
    if (backend == L"cpu") return {L"CPU", Tone::observed};
    if (backend == L"directml") return {L"DirectML GPU", Tone::observed};
    if (backend == L"cuda") return {L"CUDA GPU", Tone::observed};
    if (backend == L"rocm") return {L"ROCm GPU", Tone::observed};
    if (backend == L"mps") return {L"MPS GPU", Tone::observed};
    if (backend == L"xpu") return {L"XPU GPU", Tone::observed};
    return {};
}

inline Status ledger(std::wstring_view state, std::wstring_view positions) {
    if (state == L"invalid") return {L"Integrity failure", Tone::critical};
    if (state == L"clear") {
        if (positions != L"0") return {};
        return {L"Clear - 0 open", Tone::observed};
    }
    if (state == L"tracked") {
        std::uint64_t count = 0;
        if (!unsigned_count(positions, count) || count == 0) return {};
        return {std::wstring(positions) + L" tracked", Tone::caution};
    }
    return {};
}

inline Status runtime(std::wstring_view state) {
    if (state == L"running") return {L"Running state", Tone::observed};
    if (state == L"stopped") return {L"Stopped state", Tone::observed};
    if (state == L"paused") return {L"Paused", Tone::caution};
    if (state == L"stopping") return {L"Stopping", Tone::caution};
    if (state == L"stop_requested") return {L"Stop latched", Tone::caution};
    if (state == L"unconfigured") return {L"Unconfigured", Tone::unknown};
    if (state == L"control_unavailable") return {L"Control unavailable", Tone::critical};
    return {};
}

}  // namespace simple_ai_trading::presentation
