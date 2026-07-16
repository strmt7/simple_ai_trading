#include "command_contract.hpp"

#include <dwmapi.h>
#include <shellapi.h>
#include <windows.h>
#include <windowsx.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cwctype>
#include <exception>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace app {
using simple_ai_trading::native_contract::CommandSpec;
using simple_ai_trading::native_contract::kCommandCount;
using simple_ai_trading::native_contract::kCommandContractSha256;
using simple_ai_trading::native_contract::kCommands;
using simple_ai_trading::native_contract::kWorkflowCommandCount;
using simple_ai_trading::native_contract::kWorkflowCommands;

constexpr int kInitialWidth = 1680;
constexpr int kInitialHeight = 1020;
constexpr int kMinWidth = 1280;
constexpr int kMinHeight = 860;
constexpr COLORREF kBg = RGB(18, 22, 25);
constexpr COLORREF kShell = RGB(24, 30, 34);
constexpr COLORREF kPanel = RGB(31, 38, 43);
constexpr COLORREF kPanel2 = RGB(40, 49, 55);
constexpr COLORREF kAccent = RGB(69, 168, 151);
constexpr COLORREF kDanger = RGB(139, 55, 60);
constexpr COLORREF kText = RGB(238, 242, 244);
constexpr COLORREF kMuted = RGB(172, 183, 190);
constexpr COLORREF kSubtle = RGB(115, 127, 135);
constexpr UINT_PTR kApiBudgetTimerId = 301;
constexpr UINT kApiBudgetRefreshMs = 90000;

enum ControlId : int {
    kPageListId = 100,
    kCommandComboId = 101,
    kArgsEditId = 102,
    kOutputEditId = 103,
    kRunSelectedId = 104,
    kSelectedHelpId = 105,
    kStopAllId = 106,
    kAiPreflightId = 107,
    kRiskReportId = 108,
    kModelLabId = 109,
    kBacktestChartId = 110,
    kStatusBarId = 111,
    kProfileComboId = 112,
    kLeverageComboId = 113,
    kAiToggleId = 114,
    kReinvestToggleId = 115,
    kModeComboId = 116,
    kQuickBaseId = 200,
};

struct CommandEntry {
    std::wstring display;
    std::wstring command;
    int contract_index = -1;
};

struct QuickAction {
    std::wstring label;
    std::vector<std::wstring> commands;
};

struct CommandResult {
    std::wstring output;
    int exit_code = 2;
};

class MainWindow {
  public:
    int run(HINSTANCE instance, int show) {
        instance_ = instance;
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        WNDCLASSEXW wc{};
        wc.cbSize = sizeof(wc);
        wc.lpfnWndProc = &MainWindow::window_proc;
        wc.hInstance = instance;
        wc.lpszClassName = L"SimpleAITradingNativeWindow";
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wc.hbrBackground = CreateSolidBrush(kBg);
        if (!RegisterClassExW(&wc)) {
            log_startup_failure(L"RegisterClassExW");
            return 1;
        }

        RECT work_area{};
        SystemParametersInfoW(SPI_GETWORKAREA, 0, &work_area, 0);
        const int work_width = static_cast<int>(work_area.right - work_area.left);
        const int work_height = static_cast<int>(work_area.bottom - work_area.top);
        const int system_dpi = GetDpiForSystem();
        int width = std::min(MulDiv(kInitialWidth, system_dpi, 96), std::max(MulDiv(kMinWidth, system_dpi, 96), work_width - 80));
        int height = std::min(MulDiv(kInitialHeight, system_dpi, 96), std::max(MulDiv(kMinHeight, system_dpi, 96), work_height - 80));
        if (work_width > 0) {
            width = std::min(width, std::max(640, work_width - 24));
        }
        if (work_height > 0) {
            height = std::min(height, std::max(560, work_height - 24));
        }
        RECT frame{0, 0, width, height};
        AdjustWindowRectEx(&frame, WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN, FALSE, 0);
        hwnd_ = CreateWindowExW(
            0,
            wc.lpszClassName,
            L"Simple AI Trading",
            WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            frame.right - frame.left,
            frame.bottom - frame.top,
            nullptr,
            nullptr,
            instance,
            this);
        if (!hwnd_) {
            log_startup_failure(L"CreateWindowExW");
            return 1;
        }

        BOOL dark = TRUE;
        COLORREF caption = RGB(27, 32, 36);
        COLORREF caption_text = kText;
        COLORREF border = RGB(63, 83, 89);
        DwmSetWindowAttribute(hwnd_, DWMWA_USE_IMMERSIVE_DARK_MODE, &dark, sizeof(dark));
        DwmSetWindowAttribute(hwnd_, DWMWA_CAPTION_COLOR, &caption, sizeof(caption));
        DwmSetWindowAttribute(hwnd_, DWMWA_TEXT_COLOR, &caption_text, sizeof(caption_text));
        DwmSetWindowAttribute(hwnd_, DWMWA_BORDER_COLOR, &border, sizeof(border));

        ShowWindow(hwnd_, show);
        UpdateWindow(hwnd_);
        if (smoke_) {
            run_sequence({L"compute"});
        }
        MSG msg{};
        while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        return static_cast<int>(msg.wParam);
    }

  private:
    HINSTANCE instance_{};
    HWND hwnd_{};
    HWND title_{};
    HWND subtitle_{};
    HWND safety_{};
    HWND page_title_{};
    HWND page_summary_{};
    HWND page_list_{};
    HWND command_label_{};
    HWND command_combo_{};
    HWND args_label_{};
    HWND args_edit_{};
    HWND help_label_{};
    HWND quick_label_{};
    HWND tools_label_{};
    HWND output_label_{};
    HWND output_edit_{};
    HWND run_selected_{};
    HWND selected_help_{};
    HWND stop_all_{};
    HWND ai_preflight_{};
    HWND risk_report_{};
    HWND model_lab_{};
    HWND backtest_chart_{};
    HWND status_bar_{};
    HWND profile_combo_{};
    HWND leverage_combo_{};
    HWND mode_combo_{};
    HWND ai_toggle_{};
    HWND reinvest_toggle_{};
    std::array<HWND, 12> quick_buttons_{};
    HFONT title_font_{};
    HFONT body_font_{};
    HFONT small_font_{};
    HFONT mono_font_{};
    HBRUSH bg_brush_{};
    HBRUSH panel_brush_{};
    HBRUSH edit_brush_{};
    int dpi_ = 96;
    int page_index_ = 0;
    std::vector<CommandEntry> command_entries_;
    std::vector<QuickAction> quick_actions_;
    std::wstring output_{L"Ready.\r\n"};
    std::wstring api_budget_{L"API budget: loading"};
    std::mutex output_mutex_;
    std::mutex api_budget_mutex_;
    std::mutex operator_status_mutex_;
    std::atomic_bool running_{false};
    std::atomic_bool control_running_{false};
    std::atomic_bool api_budget_running_{false};
    std::atomic_bool operator_status_running_{false};
    std::atomic_bool command_contract_synced_{false};
    std::atomic_uint64_t workflow_generation_{0};
    std::wstring environment_state_{L"Environment pending"};
    std::wstring bot_state_{L"State pending"};
    std::wstring persisted_profile_{L"Conservative"};
    std::wstring persisted_leverage_{L"5x"};
    std::wstring persisted_execution_{L"Paper"};
    std::wstring compute_state_{L"Checking"};
    std::wstring ai_runtime_state_{L"unloaded"};
    std::wstring ledger_state_{L"Not checked"};
    std::wstring api_reserve_state_{L"Loading"};
    std::wstring network_state_{L"Not checked"};
    std::wstring command_contract_state_{L"Contract checking"};
    bool persisted_ai_enabled_ = true;
    bool persisted_reinvest_ = false;
    bool operator_status_initialized_ = false;
    bool ai_enabled_ = true;
    bool reinvest_enabled_ = false;
    bool smoke_ = false;
    bool dry_run_ = false;

    static constexpr std::array<const wchar_t*, 7> kPages{
        L"Overview",
        L"Trading",
        L"Research",
        L"Risk",
        L"Data",
        L"System",
        L"Settings",
    };

    static constexpr std::array<const wchar_t*, 7> kPageSummaries{
        L"Verified performance, bot-owned positions, and safety state at a glance.",
        L"Launch guarded testnet runs and inspect autonomous lifecycle state.",
        L"Train, evaluate, review, and preserve optimization evidence artifacts.",
        L"Inspect universe eligibility, audits, signals, readiness, and risk controls.",
        L"Ingest archives, audit database health, monitor rate limits, and sync data.",
        L"Inspect compute, GPU, network, API budget, and data-pipeline health.",
        L"Configure trading defaults and use the generated expert command surface.",
    };

    static void log_startup_failure(const wchar_t* stage) {
        std::array<wchar_t, MAX_PATH> temp{};
        DWORD len = GetTempPathW(static_cast<DWORD>(temp.size()), temp.data());
        std::wstring path = (len > 0 && len < temp.size()) ? std::wstring(temp.data(), len) : L".\\";
        path += L"SimpleAITradingNativeStartup.log";
        std::wofstream log(path, std::ios::app);
        log << stage << L" failed with GetLastError=" << GetLastError() << L"\n";
    }

    static LRESULT CALLBACK window_proc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) {
        MainWindow* self = nullptr;
        if (message == WM_NCCREATE) {
            auto* create = reinterpret_cast<CREATESTRUCTW*>(lparam);
            self = static_cast<MainWindow*>(create->lpCreateParams);
            self->hwnd_ = hwnd;
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
        } else {
            self = reinterpret_cast<MainWindow*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
        }
        return self ? self->handle(message, wparam, lparam) : DefWindowProcW(hwnd, message, wparam, lparam);
    }

    LRESULT handle(UINT message, WPARAM wparam, LPARAM lparam) {
        switch (message) {
        case WM_CREATE:
            on_create();
            return 0;
        case WM_SIZE:
            layout();
            InvalidateRect(hwnd_, nullptr, FALSE);
            return 0;
        case WM_GETMINMAXINFO: {
            auto* info = reinterpret_cast<MINMAXINFO*>(lparam);
            info->ptMinTrackSize.x = scale(kMinWidth);
            info->ptMinTrackSize.y = scale(kMinHeight);
            return 0;
        }
        case WM_DPICHANGED:
            dpi_ = HIWORD(wparam);
            rebuild_fonts();
            layout();
            return 0;
        case WM_COMMAND:
            on_command(LOWORD(wparam), HIWORD(wparam));
            return 0;
        case WM_TIMER:
            if (wparam == kApiBudgetTimerId) {
                refresh_api_budget_async(false);
                refresh_operator_status_async();
                return 0;
            }
            return DefWindowProcW(hwnd_, message, wparam, lparam);
        case WM_CTLCOLORSTATIC:
        case WM_CTLCOLOREDIT:
        case WM_CTLCOLORLISTBOX:
        case WM_CTLCOLORBTN:
            return color_control(reinterpret_cast<HDC>(wparam), reinterpret_cast<HWND>(lparam), message);
        case WM_MEASUREITEM:
            return measure_item(reinterpret_cast<MEASUREITEMSTRUCT*>(lparam));
        case WM_DRAWITEM:
            return draw_item(static_cast<int>(wparam), reinterpret_cast<DRAWITEMSTRUCT*>(lparam));
        case WM_PAINT:
            paint();
            return 0;
        case WM_APP + 1:
            sync_output();
            return 0;
        case WM_APP + 2:
            sync_api_budget();
            return 0;
        case WM_APP + 3:
            sync_operator_status();
            return 0;
        case WM_DESTROY:
            cleanup();
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProcW(hwnd_, message, wparam, lparam);
        }
    }

    void on_create() {
        dpi_ = GetDpiForWindow(hwnd_);
        bg_brush_ = CreateSolidBrush(kBg);
        panel_brush_ = CreateSolidBrush(kPanel);
        edit_brush_ = CreateSolidBrush(RGB(14, 18, 21));
        smoke_ = env_present(L"SIMPLE_AI_TRADING_GUI_SMOKE");
        dry_run_ = env_present(L"SIMPLE_AI_TRADING_GUI_DRY_RUN");
        rebuild_fonts();
        create_controls();
        populate_pages();
        refresh_page();
        output_ = L"Ready.\r\n" + runtime_summary() + L"\r\n";
        sync_output();
        layout();
        SetTimer(hwnd_, kApiBudgetTimerId, kApiBudgetRefreshMs, nullptr);
        refresh_api_budget_async(false);
        refresh_operator_status_async();
    }

    void cleanup() {
        KillTimer(hwnd_, kApiBudgetTimerId);
        DeleteObject(title_font_);
        DeleteObject(body_font_);
        DeleteObject(small_font_);
        DeleteObject(mono_font_);
        DeleteObject(bg_brush_);
        DeleteObject(panel_brush_);
        DeleteObject(edit_brush_);
    }

    int scale(int value) const {
        return MulDiv(value, dpi_, 96);
    }

    HFONT make_font(int dip_height, int weight = FW_NORMAL, const wchar_t* face = L"Segoe UI") const {
        return CreateFontW(
            -scale(dip_height),
            0,
            0,
            0,
            weight,
            FALSE,
            FALSE,
            FALSE,
            DEFAULT_CHARSET,
            OUT_DEFAULT_PRECIS,
            CLIP_DEFAULT_PRECIS,
            CLEARTYPE_QUALITY,
            DEFAULT_PITCH | FF_SWISS,
            face);
    }

    void rebuild_fonts() {
        if (title_font_) DeleteObject(title_font_);
        if (body_font_) DeleteObject(body_font_);
        if (small_font_) DeleteObject(small_font_);
        if (mono_font_) DeleteObject(mono_font_);
        title_font_ = make_font(24, FW_SEMIBOLD);
        body_font_ = make_font(15, FW_NORMAL);
        small_font_ = make_font(12, FW_NORMAL);
        mono_font_ = make_font(13, FW_NORMAL, L"Consolas");
        for (HWND control : all_controls()) {
            if (control) {
                SendMessageW(control, WM_SETFONT, reinterpret_cast<WPARAM>(body_font_), TRUE);
            }
        }
        if (title_) SendMessageW(title_, WM_SETFONT, reinterpret_cast<WPARAM>(title_font_), TRUE);
        if (subtitle_) SendMessageW(subtitle_, WM_SETFONT, reinterpret_cast<WPARAM>(small_font_), TRUE);
        if (safety_) SendMessageW(safety_, WM_SETFONT, reinterpret_cast<WPARAM>(body_font_), TRUE);
        if (page_title_) SendMessageW(page_title_, WM_SETFONT, reinterpret_cast<WPARAM>(title_font_), TRUE);
        if (page_summary_) SendMessageW(page_summary_, WM_SETFONT, reinterpret_cast<WPARAM>(small_font_), TRUE);
        if (output_edit_) SendMessageW(output_edit_, WM_SETFONT, reinterpret_cast<WPARAM>(mono_font_), TRUE);
    }

    std::vector<HWND> all_controls() const {
        std::vector<HWND> controls{
            title_,       subtitle_,      safety_,       page_title_,    page_summary_,  status_bar_,
            page_list_,   command_label_, command_combo_, args_label_,   args_edit_,     help_label_,
            quick_label_, tools_label_,   output_label_,
            output_edit_, run_selected_, selected_help_, stop_all_,     ai_preflight_,
            risk_report_, model_lab_,    backtest_chart_,
            profile_combo_, leverage_combo_, mode_combo_, ai_toggle_, reinvest_toggle_,
        };
        for (HWND button : quick_buttons_) {
            controls.push_back(button);
        }
        return controls;
    }

    HWND create_control(const wchar_t* klass, const wchar_t* text, DWORD style, int id, DWORD ex_style = 0) {
        HWND control = CreateWindowExW(
            ex_style,
            klass,
            text,
            WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS | style,
            0,
            0,
            10,
            10,
            hwnd_,
            reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
            instance_,
            nullptr);
        SendMessageW(control, WM_SETFONT, reinterpret_cast<WPARAM>(body_font_), TRUE);
        return control;
    }

    void create_controls() {
        title_ = create_control(L"STATIC", L"Simple AI Trading", SS_LEFT, 0);
        subtitle_ = create_control(L"STATIC", L"Day-trading workstation", SS_LEFT, 0);
        safety_ = create_control(
            L"STATIC",
            L"Conservative default: 5x futures, spot stays 1x. Testnet first.\r\nProfit reinvestment is off. Stop + Close acts only on bot-owned positions.",
            SS_LEFT | SS_NOPREFIX,
            0);
        page_title_ = create_control(L"STATIC", L"Home", SS_LEFT, 0);
        page_summary_ = create_control(L"STATIC", kPageSummaries[0], SS_LEFT | SS_NOPREFIX, 0);
        page_list_ = create_control(L"LISTBOX", L"", LBS_NOTIFY | LBS_OWNERDRAWFIXED | LBS_HASSTRINGS | WS_TABSTOP, kPageListId);
        command_label_ = create_control(L"STATIC", L"Advanced Command", SS_LEFT, 0);
        command_combo_ = create_control(
            L"COMBOBOX",
            L"",
            CBS_DROPDOWNLIST | CBS_HASSTRINGS | WS_TABSTOP | WS_VSCROLL,
            kCommandComboId);
        args_label_ = create_control(L"STATIC", L"Extra Flags", SS_LEFT, 0);
        args_edit_ = create_control(L"EDIT", L"", ES_AUTOHSCROLL | WS_TABSTOP, kArgsEditId);
        help_label_ = create_control(L"STATIC", L"", SS_LEFT | SS_NOPREFIX, 0);
        quick_label_ = create_control(L"STATIC", L"Recommended Workflows", SS_LEFT, 0);
        tools_label_ = create_control(L"STATIC", L"Safety Controls", SS_LEFT, 0);
        output_label_ = create_control(L"STATIC", L"Activity Log", SS_LEFT, 0);
        output_edit_ = create_control(
            L"EDIT",
            L"",
            ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL | WS_TABSTOP,
            kOutputEditId);
        run_selected_ = create_control(L"BUTTON", L"Run Command", BS_OWNERDRAW | WS_TABSTOP, kRunSelectedId);
        selected_help_ = create_control(L"BUTTON", L"Command Help", BS_OWNERDRAW | WS_TABSTOP, kSelectedHelpId);
        stop_all_ = create_control(L"BUTTON", L"Stop + Close", BS_OWNERDRAW | WS_TABSTOP, kStopAllId);
        ai_preflight_ = create_control(L"BUTTON", L"Pause", BS_OWNERDRAW | WS_TABSTOP, kAiPreflightId);
        risk_report_ = create_control(L"BUTTON", L"Risk Review", BS_OWNERDRAW | WS_TABSTOP, kRiskReportId);
        model_lab_ = create_control(L"BUTTON", L"Positions", BS_OWNERDRAW | WS_TABSTOP, kModelLabId);
        backtest_chart_ = create_control(L"BUTTON", L"Reconcile", BS_OWNERDRAW | WS_TABSTOP, kBacktestChartId);
        status_bar_ = create_control(L"STATIC", L"API budget: loading", SS_LEFT | SS_NOPREFIX, kStatusBarId);
        profile_combo_ = create_control(
            L"COMBOBOX",
            L"",
            CBS_DROPDOWNLIST | CBS_HASSTRINGS | WS_TABSTOP | WS_VSCROLL,
            kProfileComboId);
        leverage_combo_ = create_control(
            L"COMBOBOX",
            L"",
            CBS_DROPDOWNLIST | CBS_HASSTRINGS | WS_TABSTOP | WS_VSCROLL,
            kLeverageComboId);
        mode_combo_ = create_control(
            L"COMBOBOX",
            L"",
            CBS_DROPDOWNLIST | CBS_HASSTRINGS | WS_TABSTOP | WS_VSCROLL,
            kModeComboId);
        ai_toggle_ = create_control(
            L"BUTTON",
            L"AI on (gated)",
            BS_AUTOCHECKBOX | BS_OWNERDRAW | WS_TABSTOP,
            kAiToggleId);
        reinvest_toggle_ = create_control(
            L"BUTTON",
            L"Reinvest off",
            BS_AUTOCHECKBOX | BS_OWNERDRAW | WS_TABSTOP,
            kReinvestToggleId);
        for (const wchar_t* value : {L"Conservative", L"Regular", L"Aggressive"}) {
            SendMessageW(profile_combo_, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(value));
        }
        for (const wchar_t* value : {L"1x", L"2x", L"3x", L"5x", L"10x", L"15x", L"20x"}) {
            SendMessageW(leverage_combo_, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(value));
        }
        for (const wchar_t* value : {L"Paper", L"Testnet live"}) {
            SendMessageW(mode_combo_, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(value));
        }
        SendMessageW(profile_combo_, CB_SETCURSEL, 0, 0);
        SendMessageW(leverage_combo_, CB_SETCURSEL, 3, 0);
        SendMessageW(mode_combo_, CB_SETCURSEL, 0, 0);
        SendMessageW(ai_toggle_, BM_SETCHECK, BST_CHECKED, 0);
        SendMessageW(reinvest_toggle_, BM_SETCHECK, BST_UNCHECKED, 0);
        ai_enabled_ = true;
        reinvest_enabled_ = false;
        for (int i = 0; i < static_cast<int>(quick_buttons_.size()); ++i) {
            quick_buttons_[static_cast<std::size_t>(i)] =
                create_control(L"BUTTON", L"", BS_OWNERDRAW | WS_TABSTOP, kQuickBaseId + i);
        }
    }

    void populate_pages() {
        SendMessageW(page_list_, LB_RESETCONTENT, 0, 0);
        SendMessageW(page_list_, LB_SETITEMHEIGHT, 0, static_cast<LPARAM>(scale(42)));
        for (const wchar_t* page : kPages) {
            SendMessageW(page_list_, LB_ADDSTRING, 0, reinterpret_cast<LPARAM>(page));
        }
        SendMessageW(page_list_, LB_SETCURSEL, 0, 0);
    }

    static void set_visible(HWND control, bool visible) {
        if (control) {
            ShowWindow(control, visible ? SW_SHOW : SW_HIDE);
        }
    }

    void layout_overview(const RECT& client) {
        const int sidebar = scale(220);
        const int header_h = scale(60);
        const int footer_h = scale(64);
        const int pad = scale(20);
        const int gap = scale(14);
        const int main_left = sidebar + scale(28);
        const int right = client.right - pad;
        const int footer_top = client.bottom - footer_h;
        const int settings_top = footer_top - scale(62);

        MoveWindow(title_, scale(58), scale(13), sidebar - scale(76), scale(34), TRUE);
        MoveWindow(page_list_, scale(10), header_h + scale(18), sidebar - scale(20), footer_top - header_h - scale(36), TRUE);
        MoveWindow(status_bar_, scale(148), footer_top + scale(18), scale(190), scale(30), TRUE);

        const int stop_w = scale(150);
        const int pause_w = scale(100);
        const int start_w = scale(112);
        const int action_gap = scale(12);
        const int stop_left = right - stop_w;
        const int pause_left = stop_left - action_gap - pause_w;
        const int start_left = pause_left - action_gap - start_w;
        MoveWindow(run_selected_, start_left, header_h + scale(18), start_w, scale(48), TRUE);
        MoveWindow(ai_preflight_, pause_left, header_h + scale(18), pause_w, scale(48), TRUE);
        MoveWindow(stop_all_, stop_left, header_h + scale(18), stop_w, scale(48), TRUE);

        const int control_top = settings_top + scale(10);
        MoveWindow(mode_combo_, main_left + scale(42), control_top, scale(120), scale(220), TRUE);
        MoveWindow(profile_combo_, main_left + scale(222), control_top, scale(160), scale(220), TRUE);
        MoveWindow(leverage_combo_, main_left + scale(468), control_top, scale(120), scale(220), TRUE);
        MoveWindow(ai_toggle_, main_left + scale(630), control_top, scale(150), scale(40), TRUE);
        MoveWindow(reinvest_toggle_, main_left + scale(810), control_top, scale(180), scale(40), TRUE);

        for (HWND control : {subtitle_, safety_, page_title_, page_summary_, command_label_, command_combo_,
                             args_label_, args_edit_, help_label_, quick_label_, tools_label_, output_label_,
                             output_edit_, selected_help_, risk_report_, model_lab_, backtest_chart_}) {
            set_visible(control, false);
        }
        for (HWND button : quick_buttons_) {
            set_visible(button, false);
        }
        for (HWND control : {title_, page_list_, run_selected_, ai_preflight_, stop_all_, status_bar_,
                             profile_combo_, leverage_combo_, mode_combo_, ai_toggle_, reinvest_toggle_}) {
            set_visible(control, true);
        }
        SetWindowTextW(run_selected_, L"Start");
        SetWindowTextW(ai_preflight_, L"Pause");
        SetWindowTextW(stop_all_, L"Stop + Close");
        (void)gap;
    }

    void layout() {
        if (!hwnd_ || !title_) return;
        RECT client{};
        GetClientRect(hwnd_, &client);
        if (page_index_ == 0) {
            layout_overview(client);
            return;
        }
        for (HWND control : {title_, subtitle_, safety_, page_title_, page_summary_, page_list_, command_label_,
                             command_combo_, args_label_, args_edit_, help_label_, quick_label_, tools_label_,
                             output_label_, output_edit_, run_selected_, selected_help_, stop_all_, ai_preflight_,
                             risk_report_, model_lab_, backtest_chart_, status_bar_}) {
            set_visible(control, true);
        }
        for (HWND control : {profile_combo_, leverage_combo_, mode_combo_, ai_toggle_, reinvest_toggle_}) {
            set_visible(control, false);
        }
        SetWindowTextW(run_selected_, L"Run Command");
        SetWindowTextW(ai_preflight_, L"Pause");
        SetWindowTextW(stop_all_, L"Stop + Close");
        const int pad = scale(20);
        const int sidebar = scale(236);
        const int header_h = scale(86);
        const int footer_h = scale(68);
        const int gap = scale(18);
        const int right = client.right - pad;
        const int footer_top = client.bottom - footer_h;
        const int bottom = footer_top - pad;
        const int main_left = sidebar + gap;
        const int main_width = std::max(scale(720), right - main_left);

        MoveWindow(title_, pad + scale(48), scale(20), sidebar - scale(76), scale(34), TRUE);
        MoveWindow(subtitle_, pad + scale(48), scale(56), sidebar - scale(76), scale(24), TRUE);
        MoveWindow(safety_, main_left + scale(58), scale(22), std::max(scale(460), main_width - scale(560)), scale(54), TRUE);
        MoveWindow(page_list_, scale(14), scale(128), sidebar - scale(28), std::max(scale(300), bottom - scale(128)), TRUE);
        MoveWindow(page_title_, main_left + scale(42), header_h + scale(24), main_width - scale(42), scale(38), TRUE);
        MoveWindow(page_summary_, main_left + scale(42), header_h + scale(64), main_width - scale(42), scale(28), TRUE);

        const int command_card_top = header_h + scale(102);
        const int command_card_h = scale(136);
        const int command_inner_top = command_card_top + scale(18);
        const int action_w = scale(156);
        const int action_left = right - action_w - scale(22);
        const int field_left = main_left + scale(22);
        const int field_gap = scale(16);
        const int field_area_w = std::max(scale(500), action_left - field_left - gap);
        const int combo_w = std::max(scale(230), (field_area_w - field_gap) / 2);
        const int args_w = std::max(scale(230), field_area_w - combo_w - field_gap);
        const int args_left = field_left + combo_w + field_gap;
        MoveWindow(command_label_, field_left, command_inner_top, combo_w, scale(24), TRUE);
        MoveWindow(command_combo_, field_left, command_inner_top + scale(34), combo_w, scale(240), TRUE);
        MoveWindow(args_label_, args_left, command_inner_top, args_w, scale(24), TRUE);
        MoveWindow(args_edit_, args_left, command_inner_top + scale(34), args_w, scale(38), TRUE);
        MoveWindow(help_label_, field_left, command_inner_top + scale(82), std::max(scale(320), field_area_w), scale(36), TRUE);
        MoveWindow(run_selected_, action_left, command_inner_top + scale(6), action_w, scale(42), TRUE);
        MoveWindow(selected_help_, action_left, command_inner_top + scale(60), action_w, scale(40), TRUE);

        const int quick_label_top = command_card_top + command_card_h + scale(24);
        MoveWindow(quick_label_, main_left, quick_label_top, main_width, scale(26), TRUE);
        const int quick_top = quick_label_top + scale(34);
        const int quick_cols = main_width >= scale(1040) ? 4 : (main_width >= scale(760) ? 2 : 1);
        const int quick_gap = scale(14);
        const int quick_h = scale(76);
        const int quick_w = (main_width - (quick_gap * (quick_cols - 1))) / quick_cols;
        for (int i = 0; i < static_cast<int>(quick_buttons_.size()); ++i) {
            const int col = i % quick_cols;
            const int row = i / quick_cols;
            MoveWindow(
                quick_buttons_[static_cast<std::size_t>(i)],
                main_left + (col * (quick_w + quick_gap)),
                quick_top + (row * (quick_h + quick_gap)),
                quick_w,
                quick_h,
                TRUE);
        }

        const int visible_actions = std::min(static_cast<int>(quick_actions_.size()), static_cast<int>(quick_buttons_.size()));
        const int quick_rows = std::max(1, (visible_actions + quick_cols - 1) / quick_cols);
        const int tools_label_top = quick_top + (quick_rows * quick_h) + ((quick_rows - 1) * quick_gap) + scale(26);
        MoveWindow(tools_label_, main_left, tools_label_top, main_width, scale(26), TRUE);
        const int tools_top = tools_label_top + scale(34);
        const int tool_gap = scale(12);
        const int tool_w = (main_width - (tool_gap * 4)) / 5;
        MoveWindow(stop_all_, main_left, tools_top, tool_w, scale(62), TRUE);
        MoveWindow(ai_preflight_, main_left + (tool_w + tool_gap), tools_top, tool_w, scale(62), TRUE);
        MoveWindow(backtest_chart_, main_left + (2 * (tool_w + tool_gap)), tools_top, tool_w, scale(62), TRUE);
        MoveWindow(model_lab_, main_left + (3 * (tool_w + tool_gap)), tools_top, tool_w, scale(62), TRUE);
        MoveWindow(risk_report_, main_left + (4 * (tool_w + tool_gap)), tools_top, tool_w, scale(62), TRUE);

        const int output_top = tools_top + scale(116);
        MoveWindow(output_label_, main_left + scale(18), output_top - scale(34), main_width - scale(36), scale(28), TRUE);
        MoveWindow(output_edit_, main_left + scale(18), output_top, main_width - scale(36), std::max(scale(96), bottom - output_top), TRUE);
        MoveWindow(status_bar_, scale(28), footer_top + scale(24), client.right - scale(56), scale(30), TRUE);
    }

    void paint_overview(HDC dc, const RECT& client) {
        const int sidebar = scale(220);
        const int header_h = scale(60);
        const int footer_h = scale(64);
        const int footer_top = client.bottom - footer_h;
        const int main_left = sidebar + scale(28);
        const int right = client.right - scale(20);
        const int content_width = right - main_left;
        const int settings_top = footer_top - scale(62);

        fill_rect(dc, RECT{0, 0, sidebar, footer_top}, RGB(18, 26, 31));
        fill_rect(dc, RECT{0, 0, client.right, header_h}, RGB(15, 22, 27));
        fill_rect(dc, RECT{0, footer_top, client.right, client.bottom}, RGB(17, 25, 30));
        fill_rect(dc, RECT{0, header_h - scale(1), client.right, header_h}, RGB(48, 62, 69));
        fill_rect(dc, RECT{sidebar, header_h, sidebar + scale(1), footer_top}, RGB(48, 62, 69));
        fill_rect(dc, RECT{0, footer_top, client.right, footer_top + scale(1)}, RGB(48, 62, 69));

        RECT logo{scale(24), scale(17), scale(47), scale(40)};
        draw_simple_icon(dc, logo, RGB(67, 214, 211), 2);

        const int stop_left = right - scale(150);
        const int pause_left = stop_left - scale(112);
        const int start_left = pause_left - scale(124);
        RECT state_band{main_left, header_h + scale(18), start_left - scale(16), header_h + scale(66)};
        round_rect(dc, state_band, RGB(22, 31, 36), RGB(49, 65, 73), scale(4));
        std::wstring profile_state = combo_text(profile_combo_);
        std::wstring leverage_state = combo_text(leverage_combo_) + L" limit";
        std::wstring execution_state = combo_text(mode_combo_);
        std::wstring environment_state;
        std::wstring bot_state;
        std::wstring compute_state;
        std::wstring ai_runtime_state;
        std::wstring ledger_state;
        std::wstring api_reserve_state;
        std::wstring network_state;
        std::wstring command_contract_state;
        {
            std::lock_guard lock(operator_status_mutex_);
            environment_state = environment_state_;
            bot_state = bot_state_;
            compute_state = compute_state_;
            ai_runtime_state = ai_runtime_state_;
            ledger_state = ledger_state_;
            api_reserve_state = api_reserve_state_;
            network_state = network_state_;
            command_contract_state = command_contract_state_;
        }
        const bool ai_enabled = ai_enabled_;
        const bool ai_gpu_resident = ai_enabled && ai_runtime_state == L"gpu";
        std::wstring ai_state = L"AI off";
        if (ai_enabled) {
            if (ai_gpu_resident) {
                ai_state = L"AI GPU resident";
            } else if (ai_runtime_state == L"cpu") {
                ai_state = L"AI blocked (CPU)";
            } else if (ai_runtime_state == L"unavailable") {
                ai_state = L"AI unavailable";
            } else {
                ai_state = L"AI on (gated)";
            }
        }
        const std::array<std::wstring, 7> states{
            environment_state, bot_state, execution_state, profile_state,
            ai_state, leverage_state, command_contract_state};
        const int state_width = std::max(scale(76), static_cast<int>(state_band.right - state_band.left) / 7);
        for (int index = 0; index < static_cast<int>(states.size()); ++index) {
            RECT cell{state_band.left + index * state_width, state_band.top, state_band.left + (index + 1) * state_width, state_band.bottom};
            if (index > 0) {
                fill_rect(dc, RECT{cell.left, cell.top + scale(12), cell.left + scale(1), cell.bottom - scale(12)}, RGB(58, 72, 79));
            }
            RECT dot{cell.left + scale(14), cell.top + scale(19), cell.left + scale(22), cell.top + scale(27)};
            HBRUSH dot_brush = CreateSolidBrush(index == 4 && ai_gpu_resident ? RGB(68, 207, 137) : RGB(145, 158, 165));
            HGDIOBJ old = SelectObject(dc, dot_brush);
            Ellipse(dc, dot.left, dot.top, dot.right, dot.bottom);
            SelectObject(dc, old);
            DeleteObject(dot_brush);
            RECT text_rect{cell.left + scale(30), cell.top, cell.right - scale(8), cell.bottom};
            draw_text(dc, states[static_cast<std::size_t>(index)], text_rect, body_font_, index == 4 && ai_gpu_resident ? RGB(86, 210, 155) : RGB(222, 229, 232),
                      DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);
        }

        const int workspace_top = header_h + scale(86);
        const int available_height = std::max(scale(420), settings_top - workspace_top - scale(14));
        const int chart_height = std::max(scale(240), std::min(scale(330), available_height * 54 / 100));
        const int chart_width = content_width * 64 / 100;
        RECT chart{main_left, workspace_top, main_left + chart_width, workspace_top + chart_height};
        RECT gates{chart.right + scale(14), workspace_top, right, chart.bottom};
        round_rect(dc, chart, RGB(20, 29, 34), RGB(48, 64, 72), scale(4));
        round_rect(dc, gates, RGB(20, 29, 34), RGB(48, 64, 72), scale(4));

        draw_text(dc, L"Verified performance", RECT{chart.left + scale(16), chart.top + scale(12), chart.right - scale(16), chart.top + scale(38)},
                  body_font_, kText, DT_LEFT | DT_SINGLELINE);
        RECT plot{chart.left + scale(64), chart.top + scale(62), chart.right - scale(20), chart.bottom - scale(38)};
        for (int line = 0; line <= 4; ++line) {
            int y = plot.top + (plot.bottom - plot.top) * line / 4;
            fill_rect(dc, RECT{plot.left, y, plot.right, y + scale(1)}, RGB(48, 61, 68));
        }
        for (int line = 0; line <= 6; ++line) {
            int x = plot.left + (plot.right - plot.left) * line / 6;
            fill_rect(dc, RECT{x, plot.top, x + scale(1), plot.bottom}, RGB(42, 55, 62));
        }
        draw_text(dc, L"No verified run yet", plot, body_font_, kMuted, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
        draw_text(dc, L"UTC time", RECT{plot.left, plot.bottom + scale(8), plot.right, chart.bottom - scale(6)}, small_font_, kSubtle,
                  DT_CENTER | DT_SINGLELINE);

        draw_text(dc, L"Risk controls", RECT{gates.left + scale(16), gates.top + scale(12), gates.right - scale(16), gates.top + scale(38)},
                  body_font_, kText, DT_LEFT | DT_SINGLELINE);
        fill_rect(dc, RECT{gates.left, gates.top + scale(48), gates.right, gates.top + scale(49)}, RGB(48, 64, 72));
        const std::array<const wchar_t*, 5> gate_names{L"Bot ledger", L"Market regime", L"Data freshness", L"API reserve", L"Reconciliation"};
        const std::array<std::wstring, 5> gate_values{
            ledger_state, L"Not evaluated", L"Not checked", api_reserve_state, L"Not checked"};
        const int gate_row_h = std::max(scale(36), static_cast<int>(gates.bottom - gates.top - scale(49)) / 5);
        for (int index = 0; index < 5; ++index) {
            int top = gates.top + scale(49) + index * gate_row_h;
            if (index > 0) fill_rect(dc, RECT{gates.left, top, gates.right, top + scale(1)}, RGB(43, 57, 64));
            RECT name_rect{gates.left + scale(18), top, gates.left + (gates.right - gates.left) * 62 / 100, top + gate_row_h};
            RECT value_rect{name_rect.right, top, gates.right - scale(14), top + gate_row_h};
            draw_text(dc, gate_names[static_cast<std::size_t>(index)], name_rect, small_font_, kText,
                      DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);
            draw_text(dc, gate_values[static_cast<std::size_t>(index)], value_rect, small_font_, kMuted,
                      DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);
        }

        RECT positions{main_left, chart.bottom + scale(14), right, settings_top - scale(12)};
        round_rect(dc, positions, RGB(20, 29, 34), RGB(48, 64, 72), scale(4));
        draw_text(dc, L"Bot-owned positions", RECT{positions.left + scale(16), positions.top + scale(10), positions.right - scale(16), positions.top + scale(36)},
                  body_font_, kText, DT_LEFT | DT_SINGLELINE);
        const int header_top = positions.top + scale(46);
        fill_rect(dc, RECT{positions.left, header_top, positions.right, header_top + scale(1)}, RGB(48, 64, 72));
        const std::array<const wchar_t*, 8> columns{L"Symbol", L"Side", L"Size", L"Entry", L"Mark", L"Stop", L"P&L", L"Age"};
        const int column_width = (positions.right - positions.left - scale(28)) / 8;
        for (int index = 0; index < 8; ++index) {
            RECT column{positions.left + scale(14) + index * column_width, header_top + scale(8),
                        positions.left + scale(14) + (index + 1) * column_width, header_top + scale(32)};
            draw_text(dc, columns[static_cast<std::size_t>(index)], column, small_font_, kMuted,
                      DT_LEFT | DT_SINGLELINE | DT_END_ELLIPSIS);
        }
        RECT empty{positions.left + scale(20), header_top + scale(34), positions.right - scale(20), positions.bottom - scale(12)};
        draw_text(dc, L"No bot-owned positions", empty, body_font_, kMuted, DT_CENTER | DT_VCENTER | DT_SINGLELINE);

        draw_text(dc, L"Mode", RECT{main_left, settings_top + scale(18), main_left + scale(38), settings_top + scale(44)}, small_font_, kMuted,
                   DT_LEFT | DT_VCENTER | DT_SINGLELINE);
        draw_text(dc, L"Profile", RECT{main_left + scale(172), settings_top + scale(18), main_left + scale(218), settings_top + scale(44)}, small_font_, kMuted,
                   DT_LEFT | DT_VCENTER | DT_SINGLELINE);
        draw_text(dc, L"Leverage", RECT{main_left + scale(394), settings_top + scale(18), main_left + scale(464), settings_top + scale(44)}, small_font_, kMuted,
                   DT_LEFT | DT_VCENTER | DT_SINGLELINE);

        const std::array<std::wstring, 4> telemetry_names{
            L"API budget", L"Compute backend", L"Network", L"Data freshness"};
        const std::array<std::wstring, 4> telemetry_values{
            api_reserve_state, compute_state, network_state, L"Not checked"};
        const int telemetry_left = scale(28);
        const int telemetry_right = client.right - scale(118);
        const int telemetry_width = (telemetry_right - telemetry_left) / 4;
        for (int index = 0; index < 4; ++index) {
            RECT cell{telemetry_left + index * telemetry_width, footer_top, telemetry_left + (index + 1) * telemetry_width, client.bottom};
            if (index > 0) fill_rect(dc, RECT{cell.left, cell.top + scale(16), cell.left + scale(1), cell.bottom - scale(16)}, RGB(54, 69, 76));
            RECT dot{cell.left + scale(4), cell.top + scale(27), cell.left + scale(12), cell.top + scale(35)};
            const std::wstring& value = telemetry_values[static_cast<std::size_t>(index)];
            const bool confirmed = value != L"Not checked" && value != L"Checking" && value != L"Unavailable" && value != L"Loading";
            HBRUSH brush = CreateSolidBrush(confirmed ? RGB(68, 207, 137) : RGB(126, 139, 146));
            HGDIOBJ old = SelectObject(dc, brush);
            Ellipse(dc, dot.left, dot.top, dot.right, dot.bottom);
            SelectObject(dc, old);
            DeleteObject(brush);
            draw_text(dc, telemetry_names[static_cast<std::size_t>(index)],
                      RECT{cell.left + scale(20), cell.top, cell.left + telemetry_width * 62 / 100, cell.bottom}, small_font_, kText,
                      DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);
            if (index > 0) {
                draw_text(dc, value,
                          RECT{cell.left + telemetry_width * 62 / 100, cell.top, cell.right - scale(8), cell.bottom}, small_font_, kMuted,
                          DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);
            }
        }
        draw_text(dc, L"Details  >", RECT{client.right - scale(110), footer_top, client.right - scale(20), client.bottom}, small_font_, kText,
                  DT_RIGHT | DT_VCENTER | DT_SINGLELINE);
    }

    void paint() {
        PAINTSTRUCT ps{};
        HDC dc = BeginPaint(hwnd_, &ps);
        RECT client{};
        GetClientRect(hwnd_, &client);
        fill_rect(dc, client, kBg);
        if (page_index_ == 0) {
            paint_overview(dc, client);
            EndPaint(hwnd_, &ps);
            return;
        }

        const int pad = scale(20);
        const int sidebar = scale(236);
        const int header_h = scale(86);
        const int footer_h = scale(68);
        const int gap = scale(18);
        const int footer_top = client.bottom - footer_h;
        const int main_left = sidebar + gap;
        const int right = client.right - pad;
        const int main_width = std::max(scale(720), right - main_left);

        RECT sidebar_rect{0, 0, sidebar, footer_top};
        fill_rect(dc, sidebar_rect, RGB(20, 27, 32));
        RECT header_rect{sidebar, 0, client.right, header_h};
        fill_rect(dc, header_rect, RGB(18, 24, 29));
        RECT footer_rect{0, footer_top, client.right, client.bottom};
        fill_rect(dc, footer_rect, RGB(19, 27, 32));

        RECT logo{pad, scale(25), pad + scale(28), scale(53)};
        draw_simple_icon(dc, logo, RGB(60, 213, 218), 2);
        RECT shield{main_left + scale(16), scale(28), main_left + scale(42), scale(54)};
        draw_simple_icon(dc, shield, RGB(185, 196, 202), 0);

        RECT health_box{right - scale(414), scale(12), right - scale(270), scale(76)};
        RECT time_box{right - scale(254), scale(12), right - scale(122), scale(76)};
        RECT config_box{right - scale(108), scale(12), right, scale(76)};
        round_rect(dc, health_box, RGB(22, 30, 36), RGB(44, 58, 66), scale(8));
        round_rect(dc, time_box, RGB(22, 30, 36), RGB(44, 58, 66), scale(8));
        round_rect(dc, config_box, RGB(22, 30, 36), RGB(44, 58, 66), scale(8));
        RECT health_title{health_box.left + scale(16), health_box.top + scale(10), health_box.right - scale(12), health_box.top + scale(31)};
        RECT health_status{health_box.left + scale(16), health_box.top + scale(34), health_box.right - scale(12), health_box.bottom - scale(8)};
        std::wstring runtime_state;
        {
            std::lock_guard lock(operator_status_mutex_);
            runtime_state = bot_state_;
        }
        draw_text(dc, L"Operator state", health_title, small_font_, kText, DT_LEFT | DT_SINGLELINE | DT_END_ELLIPSIS);
        draw_text(dc, runtime_state, health_status, small_font_, kMuted, DT_LEFT | DT_SINGLELINE | DT_END_ELLIPSIS);
        SYSTEMTIME local_time{};
        GetLocalTime(&local_time);
        wchar_t time_value[16]{};
        swprintf_s(time_value, L"%02d:%02d:%02d", local_time.wHour, local_time.wMinute, local_time.wSecond);
        RECT time_main{time_box.left + scale(14), time_box.top + scale(10), time_box.right - scale(10), time_box.top + scale(32)};
        RECT time_caption{time_box.left + scale(14), time_box.top + scale(35), time_box.right - scale(10), time_box.bottom - scale(8)};
        draw_text(dc, time_value, time_main, body_font_, kText, DT_LEFT | DT_SINGLELINE | DT_END_ELLIPSIS);
        draw_text(dc, L"Local Time", time_caption, small_font_, kMuted, DT_LEFT | DT_SINGLELINE | DT_END_ELLIPSIS);
        RECT config_text{config_box.left + scale(14), config_box.top, config_box.right - scale(10), config_box.bottom};
        draw_text(dc, L"Configure", config_text, small_font_, kText, DT_CENTER | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);

        RECT page_icon{main_left + scale(2), header_h + scale(33), main_left + scale(28), header_h + scale(59)};
        draw_simple_icon(dc, page_icon, RGB(176, 190, 198), page_index_);

        const int command_card_top = header_h + scale(102);
        RECT command_card{main_left, command_card_top, right, command_card_top + scale(136)};
        round_rect(dc, command_card, RGB(22, 30, 36), RGB(43, 57, 65), scale(8));
        const int action_left = right - scale(156) - scale(22);
        RECT command_divider{action_left - scale(16), command_card.top + scale(18), action_left - scale(15), command_card.bottom - scale(18)};
        fill_rect(dc, command_divider, RGB(35, 45, 52));

        const int quick_label_top = command_card.bottom + scale(24);
        const int quick_top = quick_label_top + scale(34);
        const int quick_cols = main_width >= scale(1040) ? 4 : (main_width >= scale(760) ? 2 : 1);
        const int quick_gap = scale(14);
        const int quick_h = scale(76);
        const int visible_actions = std::min(static_cast<int>(quick_actions_.size()), static_cast<int>(quick_buttons_.size()));
        const int quick_rows = std::max(1, (visible_actions + quick_cols - 1) / quick_cols);
        const int tools_label_top = quick_top + (quick_rows * quick_h) + ((quick_rows - 1) * quick_gap) + scale(26);
        const int tools_top = tools_label_top + scale(34);
        const int output_top = tools_top + scale(116);
        RECT output_card{main_left, output_top - scale(46), right, footer_top - pad};
        round_rect(dc, output_card, RGB(20, 27, 32), RGB(43, 57, 65), scale(8));

        RECT footer_line{pad, footer_top, client.right - pad, footer_top + scale(1)};
        fill_rect(dc, footer_line, RGB(37, 50, 58));
        EndPaint(hwnd_, &ps);
    }

    void fill_rect(HDC dc, const RECT& rect, COLORREF color) {
        HBRUSH brush = CreateSolidBrush(color);
        FillRect(dc, &rect, brush);
        DeleteObject(brush);
    }

    void round_rect(HDC dc, const RECT& rect, COLORREF fill, COLORREF border, int radius) {
        HBRUSH brush = CreateSolidBrush(fill);
        HPEN pen = CreatePen(PS_SOLID, scale(1), border);
        HGDIOBJ old_brush = SelectObject(dc, brush);
        HGDIOBJ old_pen = SelectObject(dc, pen);
        RoundRect(dc, rect.left, rect.top, rect.right, rect.bottom, radius, radius);
        SelectObject(dc, old_brush);
        SelectObject(dc, old_pen);
        DeleteObject(brush);
        DeleteObject(pen);
    }

    void draw_text(HDC dc, const std::wstring& value, RECT rect, HFONT font, COLORREF color, UINT format) {
        SelectObject(dc, font);
        SetBkMode(dc, TRANSPARENT);
        SetTextColor(dc, color);
        DrawTextW(dc, value.c_str(), -1, &rect, format | DT_NOPREFIX);
    }

    LRESULT color_control(HDC dc, HWND control, UINT message) {
        SetTextColor(dc, kText);
        SetBkColor(dc, kBg);
        if (control == output_edit_ || control == args_edit_) {
            SetTextColor(dc, RGB(218, 228, 232));
            SetBkColor(dc, RGB(14, 18, 21));
            return reinterpret_cast<LRESULT>(edit_brush_);
        }
        if (message == WM_CTLCOLORSTATIC) {
            SetBkMode(dc, OPAQUE);
            SetBkColor(dc, kBg);
            SetTextColor(dc, control == status_bar_ ? kText : kText);
            return reinterpret_cast<LRESULT>(bg_brush_);
        }
        if (message == WM_CTLCOLORLISTBOX || control == page_list_ || control == command_combo_) {
            SetTextColor(dc, kText);
            SetBkColor(dc, kPanel);
            return reinterpret_cast<LRESULT>(panel_brush_);
        }
        if (message == WM_CTLCOLORBTN) {
            SetTextColor(dc, kText);
            SetBkColor(dc, kPanel2);
            return reinterpret_cast<LRESULT>(panel_brush_);
        }
        SetBkMode(dc, TRANSPARENT);
        return reinterpret_cast<LRESULT>(bg_brush_);
    }

    LRESULT measure_item(MEASUREITEMSTRUCT* item) {
        if (!item) {
            return FALSE;
        }
        if (item->CtlID == kPageListId) {
            item->itemHeight = scale(42);
            return TRUE;
        }
        return FALSE;
    }

    LRESULT draw_item(int id, DRAWITEMSTRUCT* item) {
        if (!item) {
            return FALSE;
        }
        if (item->CtlType == ODT_LISTBOX && id == kPageListId) {
            return draw_page_item(item);
        }
        if (item->CtlType != ODT_BUTTON) {
            return FALSE;
        }
        return draw_button_item(id, item);
    }

    LRESULT draw_page_item(DRAWITEMSTRUCT* item) {
        if (item->itemID >= kPages.size()) {
            return TRUE;
        }
        const bool selected = (item->itemState & ODS_SELECTED) != 0;
        RECT rect = item->rcItem;
        InflateRect(&rect, -scale(8), -scale(3));
        fill_rect(item->hDC, item->rcItem, kPanel);
        if (selected) {
            RECT accent{rect.left, rect.top + scale(6), rect.left + scale(3), rect.bottom - scale(6)};
            fill_rect(item->hDC, accent, RGB(60, 213, 218));
            RECT selected_rect{rect.left + scale(4), rect.top, rect.right, rect.bottom};
            round_rect(item->hDC, selected_rect, RGB(36, 46, 53), RGB(45, 58, 66), scale(8));
        }

        RECT icon{rect.left + scale(14), rect.top + scale(11), rect.left + scale(28), rect.top + scale(25)};
        draw_simple_icon(item->hDC, icon, selected ? RGB(60, 213, 218) : RGB(160, 174, 182), static_cast<int>(item->itemID));

        RECT label{rect.left + scale(42), rect.top, rect.right - scale(8), rect.bottom};
        draw_text(item->hDC, kPages[item->itemID], label, body_font_, selected ? kText : RGB(218, 226, 230), DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);
        return TRUE;
    }

    LRESULT draw_button_item(int id, DRAWITEMSTRUCT* item) {
        const bool selected = (item->itemState & ODS_SELECTED) != 0;
        const bool disabled = (item->itemState & ODS_DISABLED) != 0;
        const bool focused = (item->itemState & ODS_FOCUS) != 0;
        const bool danger = id == kStopAllId;
        const bool primary = id == kRunSelectedId;
        const bool toggle = id == kAiToggleId || id == kReinvestToggleId;
        const bool checked = id == kAiToggleId ? ai_enabled_ : (id == kReinvestToggleId ? reinvest_enabled_ : false);
        const bool workflow_card = id >= kQuickBaseId;
        const bool safety_card = id == kStopAllId || id == kAiPreflightId || id == kRiskReportId || id == kModelLabId || id == kBacktestChartId;
        const bool compact_overview_action = page_index_ == 0 && (primary || id == kStopAllId || id == kAiPreflightId);
        COLORREF fill = danger ? RGB(57, 31, 36) : (primary ? RGB(29, 86, 80) : (checked ? RGB(24, 76, 72) : RGB(28, 36, 42)));
        if (selected) {
            fill = danger ? RGB(80, 38, 43) : (primary ? RGB(38, 103, 96) : RGB(36, 46, 53));
        }
        if (disabled) {
            fill = RGB(24, 30, 34);
        }
        COLORREF border = focused ? RGB(60, 213, 218) : (danger ? RGB(169, 73, 82) : (checked ? RGB(61, 189, 180) : RGB(57, 72, 82)));
        COLORREF text = disabled ? kSubtle : kText;
        round_rect(item->hDC, item->rcItem, fill, border, scale(workflow_card || safety_card ? 8 : 4));
        RECT label = item->rcItem;
        InflateRect(&label, -scale(12), 0);
        if (selected) {
            OffsetRect(&label, scale(1), scale(1));
        }
        std::wstring text_value = edit_text(item->hwndItem);
        if (toggle) {
            RECT indicator{label.left + scale(8), label.top + scale(12), label.left + scale(34), label.bottom - scale(12)};
            round_rect(item->hDC, indicator, checked ? RGB(56, 196, 184) : RGB(48, 58, 64), checked ? RGB(102, 230, 218) : RGB(86, 99, 106), scale(12));
            if (checked) {
                HPEN mark_pen = CreatePen(PS_SOLID, scale(2), RGB(10, 45, 43));
                HGDIOBJ previous_pen = SelectObject(item->hDC, mark_pen);
                MoveToEx(item->hDC, indicator.left + scale(6), indicator.top + scale(9), nullptr);
                LineTo(item->hDC, indicator.left + scale(11), indicator.bottom - scale(6));
                LineTo(item->hDC, indicator.right - scale(5), indicator.top + scale(6));
                SelectObject(item->hDC, previous_pen);
                DeleteObject(mark_pen);
            }
            label.left += scale(44);
            draw_text(item->hDC, text_value, label, small_font_, text, DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);
        } else if ((workflow_card || safety_card) && !compact_overview_action) {
            RECT icon{label.left + scale(10), label.top + scale(14), label.left + scale(34), label.top + scale(38)};
            draw_simple_icon(item->hDC, icon, danger ? RGB(255, 92, 104) : (primary ? RGB(61, 210, 184) : RGB(67, 188, 220)), id);
            label.left += scale(54);
            draw_text(item->hDC, text_value, label, body_font_, text, DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);
        } else {
            draw_text(item->hDC, text_value, label, body_font_, text, DT_CENTER | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);
        }
        if (focused) {
            RECT focus = item->rcItem;
            InflateRect(&focus, -scale(4), -scale(4));
            DrawFocusRect(item->hDC, &focus);
        }
        return TRUE;
    }

    void draw_simple_icon(HDC dc, RECT rect, COLORREF color, int seed) {
        HPEN pen = CreatePen(PS_SOLID, scale(2), color);
        HBRUSH brush = CreateSolidBrush(RGB(25, 34, 40));
        HGDIOBJ old_pen = SelectObject(dc, pen);
        HGDIOBJ old_brush = SelectObject(dc, brush);
        const int mode = std::abs(seed) % 5;
        if (mode == 0) {
            MoveToEx(dc, rect.left + (rect.right - rect.left) / 2, rect.top, nullptr);
            LineTo(dc, rect.right, rect.top + scale(7));
            LineTo(dc, rect.right - scale(4), rect.bottom);
            LineTo(dc, rect.left + scale(4), rect.bottom);
            LineTo(dc, rect.left, rect.top + scale(7));
            LineTo(dc, rect.left + (rect.right - rect.left) / 2, rect.top);
        } else if (mode == 1) {
            Rectangle(dc, rect.left, rect.top, rect.right, rect.bottom);
            MoveToEx(dc, rect.left + scale(4), rect.top + scale(7), nullptr);
            LineTo(dc, rect.right - scale(4), rect.top + scale(7));
            MoveToEx(dc, rect.left + scale(4), rect.top + scale(14), nullptr);
            LineTo(dc, rect.right - scale(4), rect.top + scale(14));
        } else if (mode == 2) {
            MoveToEx(dc, rect.left, rect.bottom, nullptr);
            LineTo(dc, rect.left + scale(6), rect.top + scale(10));
            LineTo(dc, rect.left + scale(13), rect.top + scale(15));
            LineTo(dc, rect.right - scale(3), rect.top);
            MoveToEx(dc, rect.right - scale(4), rect.top, nullptr);
            LineTo(dc, rect.right - scale(4), rect.top + scale(8));
            MoveToEx(dc, rect.right - scale(4), rect.top, nullptr);
            LineTo(dc, rect.right - scale(12), rect.top);
        } else if (mode == 3) {
            MoveToEx(dc, rect.left + (rect.right - rect.left) / 2, rect.top, nullptr);
            LineTo(dc, rect.right, rect.bottom);
            LineTo(dc, rect.left, rect.bottom);
            LineTo(dc, rect.left + (rect.right - rect.left) / 2, rect.top);
            MoveToEx(dc, rect.left + (rect.right - rect.left) / 2, rect.top + scale(8), nullptr);
            LineTo(dc, rect.left + (rect.right - rect.left) / 2, rect.bottom - scale(5));
        } else {
            Ellipse(dc, rect.left, rect.top, rect.right, rect.bottom);
            MoveToEx(dc, rect.left + (rect.right - rect.left) / 2, rect.top + scale(4), nullptr);
            LineTo(dc, rect.left + (rect.right - rect.left) / 2, rect.bottom - scale(4));
        }
        SelectObject(dc, old_brush);
        SelectObject(dc, old_pen);
        DeleteObject(brush);
        DeleteObject(pen);
    }

    void frame_rect(HDC dc, const RECT& rect, COLORREF color) {
        HBRUSH brush = CreateSolidBrush(color);
        FrameRect(dc, &rect, brush);
        DeleteObject(brush);
    }

    void on_command(int id, int notification) {
        if (id == kPageListId && notification == LBN_SELCHANGE) {
            int next = static_cast<int>(SendMessageW(page_list_, LB_GETCURSEL, 0, 0));
            if (next >= 0 && next < static_cast<int>(kPages.size())) {
                page_index_ = next;
                refresh_page();
            }
            return;
        }
        if (id == kCommandComboId && notification == CBN_SELCHANGE) {
            update_selected_help();
            return;
        }
        if (id == kProfileComboId && notification == CBN_SELCHANGE) {
            int profile = static_cast<int>(SendMessageW(profile_combo_, CB_GETCURSEL, 0, 0));
            const int leverage_index = profile == 1 ? 4 : (profile == 2 ? 5 : 3);
            SendMessageW(leverage_combo_, CB_SETCURSEL, leverage_index, 0);
            InvalidateRect(hwnd_, nullptr, FALSE);
            return;
        }
        if (id == kModeComboId && notification == CBN_SELCHANGE) {
            InvalidateRect(hwnd_, nullptr, FALSE);
            return;
        }
        if (notification != BN_CLICKED) {
            return;
        }
        if (id == kAiToggleId) {
            ai_enabled_ = !ai_enabled_;
            SendMessageW(ai_toggle_, BM_SETCHECK, ai_enabled_ ? BST_CHECKED : BST_UNCHECKED, 0);
            SetWindowTextW(ai_toggle_, ai_enabled_ ? L"AI on (gated)" : L"AI off");
            InvalidateRect(ai_toggle_, nullptr, TRUE);
            InvalidateRect(hwnd_, nullptr, FALSE);
            return;
        }
        if (id == kReinvestToggleId) {
            bool enabled = !reinvest_enabled_;
            if (enabled && !dry_run_enabled()) {
                const int answer = MessageBoxW(
                    hwnd_,
                    L"Reinvesting profits compounds both gains and losses and increases capital at risk. Continue?",
                    L"Enable profit reinvestment",
                    MB_ICONWARNING | MB_YESNO | MB_DEFBUTTON2);
                if (answer != IDYES) {
                    SendMessageW(reinvest_toggle_, BM_SETCHECK, BST_UNCHECKED, 0);
                    enabled = false;
                }
            }
            reinvest_enabled_ = enabled;
            SendMessageW(reinvest_toggle_, BM_SETCHECK, enabled ? BST_CHECKED : BST_UNCHECKED, 0);
            SetWindowTextW(reinvest_toggle_, enabled ? L"Reinvest on" : L"Reinvest off");
            InvalidateRect(reinvest_toggle_, nullptr, TRUE);
            return;
        }
        switch (id) {
        case kRunSelectedId:
            if (page_index_ == 0) {
                run_overview_start();
            } else {
                run_selected();
            }
            return;
        case kSelectedHelpId:
            run_selected_help();
            return;
        case kStopAllId:
            {
                std::lock_guard lock(operator_status_mutex_);
                bot_state_ = L"Stop requested";
            }
            InvalidateRect(hwnd_, nullptr, FALSE);
            run_control_sequence({L"autonomous stop"});
            return;
        case kAiPreflightId:
            {
                std::lock_guard lock(operator_status_mutex_);
                bot_state_ = L"Pause requested";
            }
            InvalidateRect(hwnd_, nullptr, FALSE);
            run_control_sequence({L"autonomous pause"});
            return;
        case kRiskReportId:
            run_sequence({L"risk --paper"});
            return;
        case kModelLabId:
            run_sequence({L"positions"});
            return;
        case kBacktestChartId:
            run_sequence({L"reconcile"});
            return;
        default:
            if (id >= kQuickBaseId && id < kQuickBaseId + static_cast<int>(quick_buttons_.size())) {
                int index = id - kQuickBaseId;
                if (index >= 0 && index < static_cast<int>(quick_actions_.size())) {
                    run_sequence(quick_actions_[static_cast<std::size_t>(index)].commands);
                }
            }
            return;
        }
    }

    void refresh_page() {
        if (page_title_ && page_index_ >= 0 && page_index_ < static_cast<int>(kPages.size())) {
            SetWindowTextW(page_title_, kPages[static_cast<std::size_t>(page_index_)]);
        }
        if (page_summary_ && page_index_ >= 0 && page_index_ < static_cast<int>(kPageSummaries.size())) {
            SetWindowTextW(page_summary_, kPageSummaries[static_cast<std::size_t>(page_index_)]);
        }
        refresh_command_combo();
        refresh_quick_actions();
        update_selected_help();
        layout();
        InvalidateRect(hwnd_, nullptr, TRUE);
    }

    void refresh_command_combo() {
        command_entries_.clear();
        SendMessageW(command_combo_, CB_RESETCONTENT, 0, 0);
        if (page_index_ > 0 && page_index_ < static_cast<int>(kPages.size()) - 1) {
            const std::wstring page = kPages[static_cast<std::size_t>(page_index_)];
            for (int i = 0; i < kWorkflowCommandCount; ++i) {
                const auto& item = kWorkflowCommands[i];
                if (page == item.page) {
                    add_command_entry(item.group, item.command);
                }
            }
        } else if (page_index_ == static_cast<int>(kPages.size()) - 1) {
            for (int i = 0; i < kWorkflowCommandCount; ++i) {
                const auto& item = kWorkflowCommands[i];
                const std::wstring group = std::wstring(item.page) + L" - " + item.group;
                add_command_entry(group.c_str(), item.command);
            }
        }
        for (std::size_t i = 0; i < command_entries_.size(); ++i) {
            LRESULT item = SendMessageW(command_combo_, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(command_entries_[i].display.c_str()));
            SendMessageW(command_combo_, CB_SETITEMDATA, static_cast<WPARAM>(item), static_cast<LPARAM>(i));
        }
        if (!command_entries_.empty()) {
            SendMessageW(command_combo_, CB_SETCURSEL, 0, 0);
        }
    }

    void add_command_entry(const wchar_t* group, const wchar_t* name) {
        int contract_index = command_index(name);
        if (contract_index < 0) {
            return;
        }
        CommandEntry entry{};
        entry.display = std::wstring(group) + L" / " + name;
        entry.command = name;
        entry.contract_index = contract_index;
        command_entries_.push_back(entry);
    }

    int command_index(const std::wstring& name) const {
        for (int i = 0; i < kCommandCount; ++i) {
            if (name == kCommands[i].name) {
                return i;
            }
        }
        return -1;
    }

    void refresh_quick_actions() {
        quick_actions_.clear();
        if (page_index_ == 0) {
            quick_actions_ = {
                {L"System Check", {L"compute", L"api-budget --compact", L"doctor"}},
                {L"Paper Trial", {L"live --paper --steps 1"}},
                {L"Research Run", {L"model-lab --objective conservative --max-symbols 3 --max-scan 20 --limit 500"}},
                {L"Backtest Graph", {L"backtest-chart"}},
            };
        } else if (page_index_ == 1) {
            quick_actions_ = {
                {L"Paper Step", {L"live --paper --steps 1"}},
                {L"Guarded Start", {L"autonomous start --paper --iterations 1"}},
                {L"Run Status", {L"autonomous status"}},
                {L"Connect", {L"connect"}},
            };
        } else if (page_index_ == 2) {
            quick_actions_ = {
                {L"Conservative Lab", {L"model-lab --objective conservative --max-symbols 3 --max-scan 20 --limit 500"}},
                {L"Regular Futures Lab", {L"model-lab --objective regular --max-symbols 3 --max-scan 20 --limit 500 --market futures"}},
                {L"Aggressive Futures", {L"model-lab --objective aggressive --max-symbols 3 --max-scan 20 --limit 500 --market futures"}},
                {L"AI Review", {L"ai-review --report data/model_lab/model_lab_report.json"}},
            };
        } else if (page_index_ == 3) {
            quick_actions_ = {
                {L"Universe Gate", {L"universe"}},
                {L"Audit Trail", {L"audit"}},
                {L"Signal Health", {L"signals"}},
                {L"Source Grades", {L"source-grades"}},
            };
        } else if (page_index_ == 4) {
            quick_actions_ = {
                {L"Data Health", {L"data-health --interval 1s --market spot --json"}},
                {L"Rate Limit Detail", {L"api-budget --compact"}},
                {L"Archive Sync Help", {L"archive-sync --help"}},
                {L"Data Sync Help", {L"data-sync --help"}},
            };
        } else if (page_index_ == 5) {
            quick_actions_ = {
                {L"Compute Backend", {L"compute"}},
                {L"API Budget", {L"api-budget --compact"}},
                {L"System Doctor", {L"doctor"}},
                {L"Runtime Status", {L"status"}},
            };
        } else {
            quick_actions_ = {
                {L"Configure", {L"configure"}},
                {L"Strategy", {L"strategy --help"}},
                {L"AI Settings", {L"ai"}},
                {L"CLI Shell", {L"shell --help"}},
            };
        }
        for (int i = 0; i < static_cast<int>(quick_buttons_.size()); ++i) {
            HWND button = quick_buttons_[static_cast<std::size_t>(i)];
            if (i < static_cast<int>(quick_actions_.size())) {
                SetWindowTextW(button, quick_actions_[static_cast<std::size_t>(i)].label.c_str());
                ShowWindow(button, SW_SHOW);
                EnableWindow(button, TRUE);
            } else {
                SetWindowTextW(button, L"");
                ShowWindow(button, SW_HIDE);
            }
        }
    }

    void update_selected_help() {
        CommandEntry* entry = selected_entry();
        if (!entry) {
            SetWindowTextW(help_label_, L"No command selected.");
            return;
        }
        const CommandSpec& spec = kCommands[entry->contract_index];
        std::wstring help = spec.help;
        help += L"\r\n";
        help += option_preview(spec);
        SetWindowTextW(help_label_, help.c_str());
    }

    CommandEntry* selected_entry() {
        int sel = static_cast<int>(SendMessageW(command_combo_, CB_GETCURSEL, 0, 0));
        if (sel < 0) {
            return nullptr;
        }
        LRESULT data = SendMessageW(command_combo_, CB_GETITEMDATA, static_cast<WPARAM>(sel), 0);
        if (data == CB_ERR || data < 0 || data >= static_cast<LRESULT>(command_entries_.size())) {
            return nullptr;
        }
        return &command_entries_[static_cast<std::size_t>(data)];
    }

    static std::wstring option_preview(const CommandSpec& command) {
        if (command.option_count <= 0 || command.options == nullptr) {
            return L"No CLI options.";
        }
        std::wstring preview = L"Options: ";
        const int shown = std::min(command.option_count, 8);
        for (int i = 0; i < shown; ++i) {
            if (i > 0) {
                preview += L", ";
            }
            preview += command.options[i].flags;
            if (command.options[i].repeatable) {
                preview += L" (repeatable)";
            }
        }
        if (command.option_count > shown) {
            preview += L", ... +";
            preview += std::to_wstring(command.option_count - shown);
        }
        return preview;
    }

    static std::wstring combo_text(HWND combo) {
        const int selection = static_cast<int>(SendMessageW(combo, CB_GETCURSEL, 0, 0));
        if (selection < 0) {
            return L"";
        }
        const int length = static_cast<int>(SendMessageW(combo, CB_GETLBTEXTLEN, selection, 0));
        if (length < 0) {
            return L"";
        }
        std::wstring value(static_cast<std::size_t>(length) + 1, L'\0');
        SendMessageW(combo, CB_GETLBTEXT, selection, reinterpret_cast<LPARAM>(value.data()));
        value.resize(static_cast<std::size_t>(length));
        return value;
    }

    void run_overview_start() {
        std::wstring profile = combo_text(profile_combo_);
        std::transform(profile.begin(), profile.end(), profile.begin(), [](wchar_t value) {
            return static_cast<wchar_t>(towlower(value));
        });
        if (profile.empty()) {
            profile = L"conservative";
        }
        std::wstring leverage = combo_text(leverage_combo_);
        if (!leverage.empty() && leverage.back() == L'x') {
            leverage.pop_back();
        }
        if (leverage.empty()) {
            leverage = L"5";
        }
        const bool testnet_live = combo_text(mode_combo_) == L"Testnet live";
        if (testnet_live && !dry_run_enabled()) {
            const int answer = MessageBoxW(
                hwnd_,
                L"Testnet live mode submits real orders to the configured Binance testnet account. Continue?",
                L"Start testnet trading",
                MB_ICONWARNING | MB_YESNO | MB_DEFBUTTON2);
            if (answer != IDYES) {
                return;
            }
        }
        const bool ai_enabled = ai_enabled_;
        const bool reinvest = reinvest_enabled_;
        std::wstring strategy_command = L"strategy --profile " + profile + L" --leverage " + leverage;
        strategy_command += reinvest ? L" --reinvest-profits" : L" --no-reinvest-profits";
        {
            std::lock_guard lock(operator_status_mutex_);
            bot_state_ = L"Start requested";
        }
        InvalidateRect(hwnd_, nullptr, FALSE);
        run_sequence({
            strategy_command,
            ai_enabled ? L"ai --enable" : L"ai --disable",
            L"autonomous start --objective " + profile + (testnet_live ? L" --live" : L" --paper"),
        });
    }

    void run_selected() {
        CommandEntry* entry = selected_entry();
        if (!entry) {
            return;
        }
        std::wstring command = entry->command;
        std::wstring extra = edit_text(args_edit_);
        extra = trim(extra);
        if (!extra.empty()) {
            command += L" ";
            command += extra;
        }
        run_sequence({command});
    }

    void run_selected_help() {
        CommandEntry* entry = selected_entry();
        if (!entry) {
            return;
        }
        run_sequence({entry->command + L" --help"});
    }

    static std::wstring edit_text(HWND control) {
        int len = GetWindowTextLengthW(control);
        std::wstring value(static_cast<std::size_t>(len) + 1, L'\0');
        if (len > 0) {
            GetWindowTextW(control, value.data(), len + 1);
        }
        value.resize(static_cast<std::size_t>(len));
        return value;
    }

    static std::wstring trim(const std::wstring& value) {
        const auto first = value.find_first_not_of(L" \t\r\n");
        if (first == std::wstring::npos) {
            return L"";
        }
        const auto last = value.find_last_not_of(L" \t\r\n");
        return value.substr(first, last - first + 1);
    }

    void run_sequence(std::vector<std::wstring> commands) {
        if (commands.empty()) {
            run_selected_help();
            return;
        }
        if (!smoke_ && !command_contract_synced_.load()) {
            append_output(
                L"\r\nWorkflow blocked: the native app and Python backend command contracts "
                L"are not verified as identical. Pause and Stop remain available.\r\n");
            return;
        }
        if (running_.exchange(true)) {
            append_output(L"\r\nA command is already running.\r\n");
            return;
        }
        const std::uint64_t generation = workflow_generation_.fetch_add(1) + 1;
        EnableWindow(run_selected_, FALSE);
        SetWindowTextW(output_label_, L"Output - running");
        std::thread([this, commands = std::move(commands), generation] {
            for (const std::wstring& command : commands) {
                if (workflow_generation_.load() != generation) {
                    append_output(
                        L"\r\nWorkflow cancelled by a safety control. No later command was started.\r\n");
                    break;
                }
                append_output(L"\r\n> simple-ai-trading " + command + L"\r\n");
                CommandResult result = execute_cli(command);
                append_output(result.output);
                if (result.exit_code != 0) {
                    append_output(
                        L"\r\nWorkflow stopped after failed command (exit " +
                        std::to_wstring(result.exit_code) + L"). No later command was started.\r\n");
                    break;
                }
            }
            running_ = false;
            refresh_api_budget_async(true);
            refresh_operator_status_async();
            if (smoke_) {
                write_smoke_log();
            }
            PostMessageW(hwnd_, WM_APP + 1, 0, 0);
            if (smoke_) {
                PostMessageW(hwnd_, WM_CLOSE, 0, 0);
            }
        }).detach();
    }

    void run_control_sequence(std::vector<std::wstring> commands) {
        workflow_generation_.fetch_add(1);
        if (commands.empty()) {
            return;
        }
        if (control_running_.exchange(true)) {
            append_output(L"\r\nA safety control is already being processed.\r\n");
            return;
        }
        std::thread([this, commands = std::move(commands)] {
            for (const std::wstring& command : commands) {
                append_output(L"\r\n> simple-ai-trading " + command + L"\r\n");
                CommandResult result = execute_cli(command);
                append_output(result.output);
                if (result.exit_code != 0) {
                    append_output(
                        L"\r\nSafety control failed (exit " +
                        std::to_wstring(result.exit_code) +
                        L"); remaining safety controls will still be attempted.\r\n");
                }
            }
            control_running_ = false;
            refresh_api_budget_async(true);
            refresh_operator_status_async();
            PostMessageW(hwnd_, WM_APP + 1, 0, 0);
        }).detach();
    }

    void append_output(const std::wstring& text) {
        {
            std::lock_guard lock(output_mutex_);
            output_ += text;
            constexpr std::size_t kMaxOutput = 120000;
            if (output_.size() > kMaxOutput) {
                output_ = L"...\r\n" + output_.substr(output_.size() - kMaxOutput);
            }
        }
        PostMessageW(hwnd_, WM_APP + 1, 0, 0);
    }

    void sync_output() {
        std::wstring snapshot;
        {
            std::lock_guard lock(output_mutex_);
            snapshot = output_;
        }
        SetWindowTextW(output_edit_, snapshot.c_str());
        SendMessageW(output_edit_, EM_SETSEL, static_cast<WPARAM>(-1), static_cast<LPARAM>(-1));
        SendMessageW(output_edit_, EM_SCROLLCARET, 0, 0);
        EnableWindow(run_selected_, !running_);
        SetWindowTextW(output_label_, running_ ? L"Activity Log - running" : L"Activity Log");
    }

    void refresh_api_budget_async(bool cached_only) {
        if (api_budget_running_.exchange(true)) {
            return;
        }
        std::thread([this, cached_only] {
            std::wstring command = cached_only ? L"api-budget --compact --cached-only" : L"api-budget --compact";
            std::wstring text = execute_cli_first_line(command);
            if (text.empty()) {
                text = L"API budget: unavailable";
            }
            std::wstring reserve = L"Unavailable";
            const std::size_t remaining_at = text.find(L"remaining=");
            if (remaining_at != std::wstring::npos) {
                const std::size_t value_start = remaining_at + std::wstring(L"remaining=").size();
                const std::size_t value_end = text.find(L' ', value_start);
                reserve = text.substr(value_start, value_end == std::wstring::npos ? std::wstring::npos : value_end - value_start);
                reserve += L" remaining";
            } else if (text.find(L"dry-run") != std::wstring::npos) {
                reserve = L"Dry-run";
            }
            {
                std::lock_guard lock(api_budget_mutex_);
                api_budget_ = text;
            }
            {
                std::lock_guard lock(operator_status_mutex_);
                api_reserve_state_ = reserve;
                if (!cached_only) {
                    network_state_ = remaining_at == std::wstring::npos ? L"Unavailable" : L"Exchange queried";
                }
            }
            api_budget_running_ = false;
            PostMessageW(hwnd_, WM_APP + 2, 0, 0);
        }).detach();
    }

    void sync_api_budget() {
        std::wstring snapshot;
        {
            std::lock_guard lock(api_budget_mutex_);
            snapshot = api_budget_;
        }
        if (page_index_ == 0) {
            std::lock_guard lock(operator_status_mutex_);
            snapshot = api_reserve_state_;
        }
        SetWindowTextW(status_bar_, snapshot.c_str());
        InvalidateRect(hwnd_, nullptr, FALSE);
    }

    static std::wstring compact_status_value(const std::wstring& line, const std::wstring& key) {
        const std::wstring prefix = key + L"=";
        std::size_t start = line.find(prefix);
        if (start == std::wstring::npos) {
            return L"";
        }
        start += prefix.size();
        const std::size_t end = line.find(L' ', start);
        return line.substr(start, end == std::wstring::npos ? std::wstring::npos : end - start);
    }

    static std::wstring display_token(std::wstring value) {
        if (!value.empty()) {
            value.front() = static_cast<wchar_t>(towupper(value.front()));
        }
        return value;
    }

    void refresh_operator_status_async() {
        if (operator_status_running_.exchange(true)) {
            return;
        }
        std::thread([this] {
            const std::wstring line = execute_cli_first_line(L"status --compact");
            const std::wstring compute_line = execute_cli_first_line(L"compute");
            const std::wstring environment = compact_status_value(line, L"environment");
            const std::wstring state = compact_status_value(line, L"bot_state");
            const std::wstring profile = compact_status_value(line, L"risk");
            const std::wstring leverage = compact_status_value(line, L"leverage");
            const std::wstring ai = compact_status_value(line, L"ai");
            const std::wstring ai_runtime = compact_status_value(line, L"ai_runtime");
            const std::wstring reinvest = compact_status_value(line, L"reinvest");
            const std::wstring execution = compact_status_value(line, L"execution");
            const std::wstring positions = compact_status_value(line, L"positions");
            const std::wstring ledger = compact_status_value(line, L"ledger");
            const std::wstring ui_contract = compact_status_value(line, L"ui_contract");
            const std::wstring compute = compact_status_value(compute_line, L"compute");
            const bool command_contract_synced = ui_contract == kCommandContractSha256;
            {
                std::lock_guard lock(operator_status_mutex_);
                environment_state_ = environment.empty() ? L"Environment unavailable" : display_token(environment);
                bot_state_ = state.empty() ? L"State unavailable" : L"Bot " + display_token(state);
                if (!profile.empty()) persisted_profile_ = display_token(profile);
                if (!leverage.empty()) persisted_leverage_ = leverage + L"x";
                if (!ai.empty()) persisted_ai_enabled_ = ai == L"enabled";
                ai_runtime_state_ = ai_runtime.empty() ? L"unavailable" : ai_runtime;
                if (!reinvest.empty()) persisted_reinvest_ = reinvest == L"on";
                if (!execution.empty()) persisted_execution_ = execution == L"live" ? L"Testnet live" : L"Paper";
                if (ledger == L"invalid") {
                    ledger_state_ = L"Integrity failure";
                } else if (ledger == L"tracked") {
                    ledger_state_ = (positions.empty() ? L"Open positions" : positions + L" tracked");
                } else if (ledger == L"clear") {
                    ledger_state_ = L"Clear - 0 open";
                } else {
                    ledger_state_ = L"Unavailable";
                }
                if (compute.empty()) {
                    compute_state_ = L"Unavailable";
                } else if (compute == L"cpu") {
                    compute_state_ = L"CPU";
                } else if (compute == L"directml") {
                    compute_state_ = L"DirectML GPU";
                } else if (compute == L"cuda") {
                    compute_state_ = L"CUDA GPU";
                } else if (compute == L"rocm") {
                    compute_state_ = L"ROCm GPU";
                } else if (compute == L"mps") {
                    compute_state_ = L"MPS GPU";
                } else {
                    compute_state_ = display_token(compute) + L" GPU";
                }
                if (ui_contract.empty()) {
                    command_contract_state_ = L"Contract unavailable";
                } else if (command_contract_synced) {
                    command_contract_state_ = L"Contract synced";
                } else {
                    command_contract_state_ = L"Contract mismatch";
                }
            }
            command_contract_synced_.store(command_contract_synced);
            operator_status_running_ = false;
            PostMessageW(hwnd_, WM_APP + 3, 0, 0);
        }).detach();
    }

    void sync_operator_status() {
        std::wstring profile;
        std::wstring leverage;
        std::wstring execution;
        bool ai = true;
        bool reinvest = false;
        {
            std::lock_guard lock(operator_status_mutex_);
            profile = persisted_profile_;
            leverage = persisted_leverage_;
            execution = persisted_execution_;
            ai = persisted_ai_enabled_;
            reinvest = persisted_reinvest_;
        }
        if (!operator_status_initialized_) {
            const int profile_index = static_cast<int>(SendMessageW(
                profile_combo_, CB_FINDSTRINGEXACT, static_cast<WPARAM>(-1), reinterpret_cast<LPARAM>(profile.c_str())));
            if (profile_index >= 0) SendMessageW(profile_combo_, CB_SETCURSEL, profile_index, 0);
            const int leverage_index = static_cast<int>(SendMessageW(
                leverage_combo_, CB_FINDSTRINGEXACT, static_cast<WPARAM>(-1), reinterpret_cast<LPARAM>(leverage.c_str())));
            if (leverage_index >= 0) SendMessageW(leverage_combo_, CB_SETCURSEL, leverage_index, 0);
            const int execution_index = static_cast<int>(SendMessageW(
                mode_combo_, CB_FINDSTRINGEXACT, static_cast<WPARAM>(-1), reinterpret_cast<LPARAM>(execution.c_str())));
            if (execution_index >= 0) SendMessageW(mode_combo_, CB_SETCURSEL, execution_index, 0);
            ai_enabled_ = ai;
            reinvest_enabled_ = reinvest;
            SetWindowTextW(ai_toggle_, ai_enabled_ ? L"AI on (gated)" : L"AI off");
            SetWindowTextW(reinvest_toggle_, reinvest_enabled_ ? L"Reinvest on" : L"Reinvest off");
            operator_status_initialized_ = true;
        }
        InvalidateRect(hwnd_, nullptr, FALSE);
        InvalidateRect(ai_toggle_, nullptr, TRUE);
        InvalidateRect(reinvest_toggle_, nullptr, TRUE);
    }

    CommandResult execute_cli(const std::wstring& args) {
        if (dry_run_enabled()) {
            const std::wstring delay_command =
                env_string(L"SIMPLE_AI_TRADING_GUI_DRY_RUN_DELAY_COMMAND");
            if (
                (!delay_command.empty() && args == delay_command) ||
                (delay_command.empty() && args.rfind(L"autonomous start", 0) == 0)) {
                const std::wstring delay_text = env_string(L"SIMPLE_AI_TRADING_GUI_DRY_RUN_DELAY_MS");
                if (!delay_text.empty()) {
                    try {
                        const int delay_ms = std::clamp(std::stoi(delay_text), 0, 10000);
                        Sleep(static_cast<DWORD>(delay_ms));
                    } catch (const std::exception&) {
                        // Invalid test-hook values intentionally behave like zero delay.
                    }
                }
            }
            const std::wstring failed_command =
                env_string(L"SIMPLE_AI_TRADING_GUI_DRY_RUN_FAIL_COMMAND");
            const int exit_code = !failed_command.empty() && args == failed_command ? 2 : 0;
            return {
                L"dry-run: simple-ai-trading " + args + L"\r\n\r\n(exit " +
                    std::to_wstring(exit_code) + L")\r\n",
                exit_code,
            };
        }
        std::wstring command = shell_command_for_cli(args);
        FILE* pipe = _wpopen(command.c_str(), L"r");
        std::wstring captured;
        if (!pipe) {
            return {L"Failed to launch command.\r\n\r\n(exit 2)\r\n", 2};
        }
        std::array<wchar_t, 1024> buffer{};
        while (fgetws(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
            captured += buffer.data();
        }
        int exit_code = _pclose(pipe);
        captured += L"\r\n(exit " + std::to_wstring(exit_code) + L")\r\n";
        return {std::move(captured), exit_code};
    }

    std::wstring execute_cli_first_line(const std::wstring& args) {
        if (dry_run_enabled()) {
            if (args == L"status --compact") {
                std::wstring status =
                    L"environment=testnet bot_state=stopped risk=conservative leverage=5 ai=enabled ai_runtime=gpu reinvest=off "
                    L"symbol=BTCUSDT market=futures execution=paper positions=0 ledger=clear ui_contract=";
                const std::wstring contract_override =
                    env_string(L"SIMPLE_AI_TRADING_GUI_DRY_RUN_CONTRACT_SHA256");
                status += contract_override.empty() ? kCommandContractSha256 : contract_override;
                return status;
            }
            return L"API budget: dry-run";
        }
        std::wstring command = shell_command_for_cli(args);
        FILE* pipe = _wpopen(command.c_str(), L"r");
        if (!pipe) {
            return L"";
        }
        std::array<wchar_t, 2048> buffer{};
        std::wstring first;
        if (fgetws(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
            first = trim(buffer.data());
        }
        _pclose(pipe);
        return first;
    }

    static bool env_present(const wchar_t* name) {
        std::array<wchar_t, 8> value{};
        DWORD size = GetEnvironmentVariableW(name, value.data(), static_cast<DWORD>(value.size()));
        return size > 0;
    }

    bool dry_run_enabled() const {
        return dry_run_ || env_present(L"SIMPLE_AI_TRADING_GUI_DRY_RUN");
    }

    static std::wstring env_string(const wchar_t* name) {
        DWORD size = GetEnvironmentVariableW(name, nullptr, 0);
        if (size == 0) {
            return L"";
        }
        std::wstring value(size, L'\0');
        GetEnvironmentVariableW(name, value.data(), size);
        while (!value.empty() && value.back() == L'\0') {
            value.pop_back();
        }
        return value;
    }

    static std::filesystem::path module_dir() {
        std::array<wchar_t, MAX_PATH> path{};
        DWORD len = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
        if (len == 0 || len >= path.size()) {
            return std::filesystem::current_path();
        }
        return std::filesystem::path(std::wstring(path.data(), len)).parent_path();
    }

    static bool looks_like_repo(const std::filesystem::path& candidate) {
        return std::filesystem::exists(candidate / L"src" / L"simple_ai_trading" / L"__init__.py");
    }

    static std::filesystem::path repo_root() {
        std::wstring env_root = env_string(L"SIMPLE_AI_TRADING_REPO_ROOT");
        if (!env_root.empty()) {
            std::filesystem::path candidate(env_root);
            if (looks_like_repo(candidate)) {
                return candidate;
            }
        }
        std::vector<std::filesystem::path> starts;
        starts.push_back(module_dir());
        starts.push_back(std::filesystem::current_path());
        for (std::filesystem::path start : starts) {
            for (int depth = 0; depth < 8 && !start.empty(); ++depth) {
                if (looks_like_repo(start)) {
                    return start;
                }
                start = start.parent_path();
            }
        }
        return {};
    }

    static std::wstring cmd_quote(std::wstring value) {
        std::wstring quoted = L"\"";
        for (wchar_t ch : value) {
            quoted += ch == L'"' ? L' ' : ch;
        }
        quoted += L"\"";
        return quoted;
    }

    static std::wstring python_invocation(const std::filesystem::path& root) {
        std::wstring env_python = env_string(L"SIMPLE_AI_TRADING_PYTHON");
        if (!env_python.empty()) {
            return cmd_quote(env_python);
        }
        if (!root.empty()) {
            std::array<std::filesystem::path, 2> candidates{
                root / L".venv311" / L"Scripts" / L"python.exe",
                root / L".venv" / L"Scripts" / L"python.exe",
            };
            for (const auto& candidate : candidates) {
                if (std::filesystem::exists(candidate)) {
                    return cmd_quote(candidate.wstring());
                }
            }
        }
        return L"py -3.11";
    }

    static std::wstring runtime_summary() {
        std::filesystem::path root = repo_root();
        std::wstring summary = L"Runtime: repo=";
        summary += root.empty() ? L"<not found>" : root.wstring();
        summary += L"; python=";
        summary += python_invocation(root);
        return summary;
    }

    static std::wstring shell_command_for_cli(const std::wstring& args) {
        std::filesystem::path root = repo_root();
        std::wstring command = L"cmd.exe /d /s /c \"";
        if (!root.empty()) {
            std::wstring root_text = root.wstring();
            command += L"cd /d " + cmd_quote(root_text) + L" && ";
            command += L"set \"SIMPLE_AI_TRADING_REPO_ROOT=" + root_text + L"\" && ";
            command += L"set \"PYTHONUTF8=1\" && ";
            command += L"set \"PYTHONPATH=" + root_text + L"\\src;%PYTHONPATH%\" && ";
        }
        command += python_invocation(root) + L" -m simple_ai_trading " + args + L" 2>&1\"";
        return command;
    }

    void write_smoke_log() {
        std::wstring path = env_string(L"SIMPLE_AI_TRADING_GUI_SMOKE_LOG");
        if (path.empty()) {
            std::array<wchar_t, MAX_PATH> temp{};
            DWORD len = GetTempPathW(static_cast<DWORD>(temp.size()), temp.data());
            path = (len > 0 && len < temp.size()) ? std::wstring(temp.data(), len) : L".\\";
            path += L"SimpleAITradingGuiSmoke.log";
        }
        std::wstring snapshot;
        {
            std::lock_guard lock(output_mutex_);
            snapshot = output_;
        }
        std::wofstream log(path, std::ios::trunc);
        log << snapshot;
    }
};
} // namespace app

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int show) {
    app::MainWindow window;
    return window.run(instance, show);
}
