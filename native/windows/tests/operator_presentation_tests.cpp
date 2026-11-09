#include "operator_presentation.hpp"

#include <iostream>
#include <string_view>
#include <vector>

namespace view = simple_ai_trading::presentation;

int main() {
    struct BudgetCase {
        std::wstring_view input;
        std::wstring_view label;
        view::Tone tone;
    };
    const std::vector<BudgetCase> cases{
        {L"dry-run: simple-ai-trading api-budget --compact", L"Dry-run", view::Tone::simulation},
        {L"API budget: ok REQUEST_WEIGHT remaining=960/1200 used=240 window=1M", L"960/1200 remaining", view::Tone::observed},
        {L"API budget: warn ORDERS remaining=5/50 used=45 window=10S", L"Low: 5/50 remaining", view::Tone::caution},
        {L"API budget: critical REQUEST_WEIGHT remaining=0/1200 used=1200 window=1M", L"Critical: 0/1200 remaining", view::Tone::critical},
        {L"API budget: ok REQUEST_WEIGHT remaining=0/1200", L"Critical: 0/1200 remaining", view::Tone::critical},
        {L"API budget: blocked retry_after=2.5s", L"Blocked", view::Tone::critical},
        {L"API budget: critical used weight unknown", L"Critical", view::Tone::critical},
        {L"API budget: warn used weight unknown", L"Low reserve", view::Tone::caution},
        {L"API budget: critical remaining=NaN", L"Critical", view::Tone::critical},
        {L"API budget: warn remaining=NaN", L"Low reserve", view::Tone::caution},
    };
    int failures = 0;
    auto check = [&failures](bool condition, std::wstring_view description) {
        if (!condition) {
            ++failures;
            std::wcerr << L"FAIL: " << description << L'\n';
        }
    };
    for (const auto& c : cases) {
        const auto result = view::api_budget(c.input);
        check(result.label == c.label && result.tone == c.tone, c.input);
    }
    // Unknown/malformed/future protocol values must never become observed health.
    for (const auto input : {
             L"", L"API budget: no sample", L"API budget: unavailable",
             L"not an API budget: ok remaining=50/100", L"API budget: future remaining=50/100",
             L"API budget: unknown remaining=50/100", L"API budget: ok used weight unknown",
             L"API budget: ok remaining=1", L"API budget: ok remaining=1/0",
             L"API budget: ok remaining=2/1", L"API budget: ok remaining=-1/100",
             L"API budget: ok remaining=1/+100", L"API budget: ok remaining=1.0/100",
             L"API budget: ok remaining=/100", L"API budget: ok remaining=1/",
             L"API budget: ok remaining=1/100/200", L"API budget: ok notremaining=50/100",
             L"API budget: ok remaining=18446744073709551616/18446744073709551616"}) {
        const auto result = view::api_budget(input);
        check(result.tone == view::Tone::unknown && result.label == L"Unavailable", input);
    }
    for (const auto backend : {L"cpu", L"directml", L"cuda", L"rocm", L"mps", L"xpu"}) {
        check(view::compute(backend).tone == view::Tone::observed, backend);
    }
    for (const auto backend : {L"", L"auto", L"unknown", L"unavailable", L"new_gpu", L"CUDA"}) {
        check(view::compute(backend).tone == view::Tone::unknown, backend);
    }
    check(view::ledger(L"invalid", L"0").tone == view::Tone::critical, L"ledger integrity failure");
    check(view::ledger(L"clear", L"0").tone == view::Tone::observed, L"clear ledger");
    check(view::ledger(L"clear", L"2").tone == view::Tone::unknown, L"contradictory clear ledger");
    check(view::ledger(L"clear", L"").tone == view::Tone::unknown, L"missing clear ledger count");
    const auto tracked = view::ledger(L"tracked", L"2");
    check(tracked.label == L"2 tracked" && tracked.tone == view::Tone::caution, L"open obligations");
    for (const auto count : {L"", L"0", L"-1", L"1.0", L"18446744073709551616"}) {
        check(view::ledger(L"tracked", count).tone == view::Tone::unknown, count);
    }
    check(view::ledger(L"future", L"1").tone == view::Tone::unknown, L"future ledger protocol");
    std::uint64_t maximum = 0;
    check(view::unsigned_count(L"18446744073709551615", maximum) &&
              maximum == std::numeric_limits<std::uint64_t>::max(), L"uint64 boundary");
    return failures == 0 ? 0 : 1;
}
