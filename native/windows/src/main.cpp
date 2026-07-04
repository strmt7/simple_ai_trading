#include "command_contract.hpp"

#include <dwmapi.h>
#include <shellapi.h>
#include <windows.h>
#include <windowsx.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace app {
using simple_ai_trading::native_contract::CommandSpec;
using simple_ai_trading::native_contract::kCommandCount;
using simple_ai_trading::native_contract::kCommands;

constexpr int kInitialWidth = 1500;
constexpr int kInitialHeight = 960;
constexpr int kMinWidth = 1180;
constexpr int kMinHeight = 840;
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
        AdjustWindowRectEx(&frame, WS_OVERLAPPEDWINDOW, FALSE, 0);
        hwnd_ = CreateWindowExW(
            0,
            wc.lpszClassName,
            L"Simple AI Trading",
            WS_OVERLAPPEDWINDOW,
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
    std::atomic_bool running_{false};
    std::atomic_bool api_budget_running_{false};
    bool smoke_ = false;
    bool dry_run_ = false;

    static constexpr std::array<const wchar_t*, 7> kPages{
        L"Dashboard",
        L"Trading",
        L"Model Lab",
        L"Risk",
        L"Market Data",
        L"Settings",
        L"All Commands",
    };

    static constexpr std::array<const wchar_t*, 7> kPageSummaries{
        L"Health, budget, risk, positions, model lab.",
        L"Paper/live controls with pause, stop, reconcile, positions, and close.",
        L"Research, train, evaluate, review, and preserve evidence artifacts.",
        L"Risk controls, universe eligibility, audits, reports, and signals.",
        L"Archive ingestion, data-health gates, API budget, fetch, and sync.",
        L"AI runtime, compute backend, strategy, configuration, and shell.",
        L"Every generated CLI command for parity and advanced operations.",
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
        title_font_ = make_font(21, FW_SEMIBOLD);
        body_font_ = make_font(13, FW_NORMAL);
        small_font_ = make_font(11, FW_NORMAL);
        mono_font_ = make_font(12, FW_NORMAL, L"Consolas");
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
            WS_CHILD | WS_VISIBLE | style,
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
            L"Conservative default. Testnet first.\r\nNo leverage or profit reinvestment. Stop closes bot-owned positions.",
            SS_LEFT | SS_NOPREFIX,
            0);
        page_title_ = create_control(L"STATIC", L"Dashboard", SS_LEFT, 0);
        page_summary_ = create_control(L"STATIC", kPageSummaries[0], SS_LEFT | SS_NOPREFIX, 0);
        page_list_ = create_control(L"LISTBOX", L"", LBS_NOTIFY | LBS_OWNERDRAWFIXED | LBS_HASSTRINGS | WS_TABSTOP, kPageListId);
        command_label_ = create_control(L"STATIC", L"Workflow Command", SS_LEFT, 0);
        command_combo_ = create_control(
            L"COMBOBOX",
            L"",
            CBS_DROPDOWNLIST | CBS_HASSTRINGS | WS_TABSTOP | WS_VSCROLL,
            kCommandComboId);
        args_label_ = create_control(L"STATIC", L"Command Options", SS_LEFT, 0);
        args_edit_ = create_control(L"EDIT", L"", ES_AUTOHSCROLL | WS_TABSTOP, kArgsEditId);
        help_label_ = create_control(L"STATIC", L"", SS_LEFT | SS_NOPREFIX, 0);
        quick_label_ = create_control(L"STATIC", L"Primary Workflows", SS_LEFT, 0);
        tools_label_ = create_control(L"STATIC", L"Safety Tools", SS_LEFT, 0);
        output_label_ = create_control(L"STATIC", L"Activity Log", SS_LEFT, 0);
        output_edit_ = create_control(
            L"EDIT",
            L"",
            ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL | WS_TABSTOP,
            kOutputEditId);
        run_selected_ = create_control(L"BUTTON", L"Run Selected", BS_OWNERDRAW | WS_TABSTOP, kRunSelectedId);
        selected_help_ = create_control(L"BUTTON", L"Show Help", BS_OWNERDRAW | WS_TABSTOP, kSelectedHelpId);
        stop_all_ = create_control(L"BUTTON", L"Stop Trading", BS_OWNERDRAW | WS_TABSTOP, kStopAllId);
        ai_preflight_ = create_control(L"BUTTON", L"AI Check", BS_OWNERDRAW | WS_TABSTOP, kAiPreflightId);
        risk_report_ = create_control(L"BUTTON", L"Risk Check", BS_OWNERDRAW | WS_TABSTOP, kRiskReportId);
        model_lab_ = create_control(L"BUTTON", L"Model Lab", BS_OWNERDRAW | WS_TABSTOP, kModelLabId);
        backtest_chart_ = create_control(L"BUTTON", L"Backtest Chart", BS_OWNERDRAW | WS_TABSTOP, kBacktestChartId);
        status_bar_ = create_control(L"STATIC", L"API budget: loading", SS_LEFT | SS_NOPREFIX, kStatusBarId);
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

    void layout() {
        if (!hwnd_ || !title_) return;
        RECT client{};
        GetClientRect(hwnd_, &client);
        const int pad = scale(20);
        const int sidebar = scale(220);
        const int header_h = scale(82);
        const int footer_h = scale(68);
        const int gap = scale(18);
        const int right = client.right - pad;
        const int footer_top = client.bottom - footer_h;
        const int bottom = footer_top - pad;
        const int main_left = sidebar + gap;
        const int main_width = std::max(scale(720), right - main_left);

        MoveWindow(title_, pad + scale(48), scale(22), sidebar - scale(76), scale(30), TRUE);
        MoveWindow(subtitle_, pad + scale(48), scale(54), sidebar - scale(76), scale(22), TRUE);
        MoveWindow(safety_, main_left + scale(58), scale(23), std::max(scale(420), main_width - scale(540)), scale(48), TRUE);
        MoveWindow(page_list_, scale(14), scale(126), sidebar - scale(28), std::max(scale(260), bottom - scale(126)), TRUE);
        MoveWindow(page_title_, main_left + scale(42), header_h + scale(26), main_width - scale(42), scale(34), TRUE);
        MoveWindow(page_summary_, main_left + scale(42), header_h + scale(62), main_width - scale(42), scale(24), TRUE);

        const int command_card_top = header_h + scale(94);
        const int command_card_h = scale(116);
        const int command_inner_top = command_card_top + scale(18);
        const int run_w = scale(142);
        const int combo_w = std::min(scale(330), std::max(scale(260), main_width * 30 / 100));
        const int args_w = std::min(scale(330), std::max(scale(240), main_width * 28 / 100));
        const int args_left = main_left + scale(22) + combo_w + gap;
        const int run_left = right - run_w - scale(22);
        const int help_left = args_left + args_w + gap;
        const int help_w = std::max(scale(220), run_left - help_left - gap);
        MoveWindow(command_label_, main_left + scale(22), command_inner_top, combo_w, scale(22), TRUE);
        MoveWindow(command_combo_, main_left + scale(22), command_inner_top + scale(32), combo_w, scale(220), TRUE);
        MoveWindow(args_label_, args_left, command_inner_top, args_w, scale(22), TRUE);
        MoveWindow(args_edit_, args_left, command_inner_top + scale(32), args_w, scale(34), TRUE);
        MoveWindow(help_label_, help_left, command_inner_top, help_w, scale(78), TRUE);
        MoveWindow(run_selected_, run_left, command_inner_top + scale(2), run_w, scale(38), TRUE);
        MoveWindow(selected_help_, run_left, command_inner_top + scale(50), run_w, scale(36), TRUE);

        const int quick_label_top = command_card_top + command_card_h + scale(24);
        MoveWindow(quick_label_, main_left, quick_label_top, main_width, scale(26), TRUE);
        const int quick_top = quick_label_top + scale(34);
        const int quick_cols = page_index_ == 0 && main_width >= scale(980) ? 5 : (main_width >= scale(780) ? 3 : 2);
        const int quick_gap = scale(12);
        const int quick_h = scale(68);
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
        const int tools_label_top = quick_top + (quick_rows * quick_h) + ((quick_rows - 1) * quick_gap) + scale(24);
        MoveWindow(tools_label_, main_left, tools_label_top, main_width, scale(26), TRUE);
        const int tools_top = tools_label_top + scale(34);
        const int tool_gap = scale(12);
        const int tool_w = (main_width - (tool_gap * 4)) / 5;
        MoveWindow(stop_all_, main_left, tools_top, tool_w, scale(58), TRUE);
        MoveWindow(ai_preflight_, main_left + (tool_w + tool_gap), tools_top, tool_w, scale(58), TRUE);
        MoveWindow(risk_report_, main_left + (2 * (tool_w + tool_gap)), tools_top, tool_w, scale(58), TRUE);
        MoveWindow(model_lab_, main_left + (3 * (tool_w + tool_gap)), tools_top, tool_w, scale(58), TRUE);
        MoveWindow(backtest_chart_, main_left + (4 * (tool_w + tool_gap)), tools_top, tool_w, scale(58), TRUE);

        const int output_top = tools_top + scale(110);
        MoveWindow(output_label_, main_left + scale(18), output_top - scale(34), main_width - scale(36), scale(28), TRUE);
        MoveWindow(output_edit_, main_left + scale(18), output_top, main_width - scale(36), std::max(scale(96), bottom - output_top), TRUE);
        MoveWindow(status_bar_, scale(28), footer_top + scale(24), client.right - scale(56), scale(30), TRUE);
    }

    void paint() {
        PAINTSTRUCT ps{};
        HDC dc = BeginPaint(hwnd_, &ps);
        RECT client{};
        GetClientRect(hwnd_, &client);
        fill_rect(dc, client, kBg);

        const int pad = scale(20);
        const int sidebar = scale(220);
        const int header_h = scale(82);
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

        RECT health_box{right - scale(372), scale(12), right - scale(242), scale(74)};
        RECT time_box{right - scale(228), scale(12), right - scale(112), scale(74)};
        RECT config_box{right - scale(98), scale(12), right, scale(74)};
        round_rect(dc, health_box, RGB(22, 30, 36), RGB(44, 58, 66), scale(8));
        round_rect(dc, time_box, RGB(22, 30, 36), RGB(44, 58, 66), scale(8));
        round_rect(dc, config_box, RGB(22, 30, 36), RGB(44, 58, 66), scale(8));
        RECT health_title{health_box.left + scale(16), health_box.top + scale(10), health_box.right - scale(12), health_box.top + scale(31)};
        RECT health_status{health_box.left + scale(16), health_box.top + scale(34), health_box.right - scale(12), health_box.bottom - scale(8)};
        draw_text(dc, L"System Health", health_title, small_font_, kText, DT_LEFT | DT_SINGLELINE | DT_END_ELLIPSIS);
        draw_text(dc, L"Healthy", health_status, small_font_, RGB(85, 206, 116), DT_LEFT | DT_SINGLELINE | DT_END_ELLIPSIS);
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

        const int command_card_top = header_h + scale(94);
        RECT command_card{main_left, command_card_top, right, command_card_top + scale(116)};
        round_rect(dc, command_card, RGB(22, 30, 36), RGB(43, 57, 65), scale(8));
        RECT command_divider{main_left + main_width * 60 / 100, command_card.top + scale(18), main_left + main_width * 60 / 100 + scale(1), command_card.bottom - scale(18)};
        fill_rect(dc, command_divider, RGB(35, 45, 52));

        const int quick_label_top = command_card.bottom + scale(24);
        const int quick_top = quick_label_top + scale(34);
        const int quick_cols = page_index_ == 0 && main_width >= scale(980) ? 5 : (main_width >= scale(780) ? 3 : 2);
        const int quick_gap = scale(12);
        const int quick_h = scale(68);
        const int visible_actions = std::min(static_cast<int>(quick_actions_.size()), static_cast<int>(quick_buttons_.size()));
        const int quick_rows = std::max(1, (visible_actions + quick_cols - 1) / quick_cols);
        const int tools_label_top = quick_top + (quick_rows * quick_h) + ((quick_rows - 1) * quick_gap) + scale(24);
        const int tools_top = tools_label_top + scale(34);
        const int output_top = tools_top + scale(110);
        RECT output_card{main_left, output_top - scale(46), right, footer_top - pad};
        round_rect(dc, output_card, RGB(20, 27, 32), RGB(43, 57, 65), scale(8));

        RECT footer_line{pad, footer_top, client.right - pad, footer_top + scale(1)};
        fill_rect(dc, footer_line, RGB(37, 50, 58));
        RECT segment{pad, footer_top + scale(18), pad + scale(130), client.bottom - scale(14)};
        draw_text(dc, L"API Budget", segment, small_font_, kMuted, DT_LEFT | DT_TOP | DT_SINGLELINE);
        RECT segment2{pad + scale(250), footer_top + scale(18), pad + scale(430), client.bottom - scale(14)};
        draw_text(dc, L"Environment\r\nTestnet", segment2, small_font_, kMuted, DT_LEFT | DT_TOP);
        RECT segment3{pad + scale(460), footer_top + scale(18), pad + scale(650), client.bottom - scale(14)};
        draw_text(dc, L"Mode\r\nAutonomous", segment3, small_font_, kMuted, DT_LEFT | DT_TOP);
        RECT segment4{pad + scale(690), footer_top + scale(18), pad + scale(900), client.bottom - scale(14)};
        draw_text(dc, L"Paper Trading\r\nEnabled", segment4, small_font_, kMuted, DT_LEFT | DT_TOP);
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
            SetBkMode(dc, TRANSPARENT);
            SetTextColor(dc, control == status_bar_ ? kText : kText);
            return reinterpret_cast<LRESULT>(GetStockObject(HOLLOW_BRUSH));
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
        const bool primary = id == kRunSelectedId || id == kModelLabId;
        const bool workflow_card = id >= kQuickBaseId;
        const bool safety_card = id == kStopAllId || id == kAiPreflightId || id == kRiskReportId || id == kModelLabId || id == kBacktestChartId;
        COLORREF fill = danger ? RGB(57, 31, 36) : (primary ? RGB(29, 86, 80) : RGB(28, 36, 42));
        if (selected) {
            fill = danger ? RGB(80, 38, 43) : (primary ? RGB(38, 103, 96) : RGB(36, 46, 53));
        }
        if (disabled) {
            fill = RGB(24, 30, 34);
        }
        COLORREF border = focused ? RGB(60, 213, 218) : (danger ? RGB(169, 73, 82) : RGB(57, 72, 82));
        COLORREF text = disabled ? kSubtle : kText;
        round_rect(item->hDC, item->rcItem, fill, border, scale(workflow_card || safety_card ? 8 : 4));
        RECT label = item->rcItem;
        InflateRect(&label, -scale(12), 0);
        if (selected) {
            OffsetRect(&label, scale(1), scale(1));
        }
        std::wstring text_value = edit_text(item->hwndItem);
        if (workflow_card || safety_card) {
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
        if (notification != BN_CLICKED) {
            return;
        }
        switch (id) {
        case kRunSelectedId:
            run_selected();
            return;
        case kSelectedHelpId:
            run_selected_help();
            return;
        case kStopAllId:
            run_sequence({L"autonomous stop", L"close all"});
            return;
        case kAiPreflightId:
            run_sequence({L"ai"});
            return;
        case kRiskReportId:
            run_sequence({L"risk --paper"});
            return;
        case kModelLabId:
            run_sequence({L"model-lab --objective conservative --max-symbols 3 --max-scan 20 --limit 500"});
            return;
        case kBacktestChartId:
            run_sequence({L"backtest-chart"});
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
    }

    void refresh_command_combo() {
        command_entries_.clear();
        SendMessageW(command_combo_, CB_RESETCONTENT, 0, 0);
        if (page_index_ == 0) {
            add_group(L"Dashboard", {L"status", L"compute", L"api-budget", L"doctor", L"positions", L"risk", L"model-lab", L"backtest-chart"});
        } else if (page_index_ == 1) {
            add_group(L"Trading", {L"connect", L"live", L"autonomous", L"positions", L"reconcile", L"close", L"spot-roundtrip"});
        } else if (page_index_ == 2) {
            add_group(L"Model Lab", {L"model-lab", L"ai-review", L"train-suite", L"train", L"prepare", L"tune", L"backtest", L"backtest-chart", L"backtest-panel", L"evaluate", L"objectives", L"signals-benchmark"});
        } else if (page_index_ == 3) {
            add_group(L"Risk", {L"risk", L"universe", L"reconcile", L"audit", L"doctor", L"signals", L"source-grades", L"report"});
        } else if (page_index_ == 4) {
            add_group(L"Market Data", {L"api-budget", L"data-health", L"archive-sync", L"data-sync", L"fetch", L"signals", L"source-grades"});
        } else if (page_index_ == 5) {
            add_group(L"Settings", {L"ai", L"compute", L"configure", L"strategy", L"menu", L"shell"});
        } else {
            for (int i = 0; i < kCommandCount; ++i) {
                add_command_entry(L"CLI", kCommands[i].name);
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

    void add_group(const wchar_t* group, std::initializer_list<const wchar_t*> names) {
        for (const wchar_t* name : names) {
            add_command_entry(group, name);
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
                {L"Health Check", {L"compute", L"api-budget --compact", L"doctor"}},
                {L"Paper Status", {L"status", L"positions"}},
                {L"Risk Snapshot", {L"risk --paper"}},
                {L"Backtest Chart", {L"backtest-chart"}},
                {L"Model Lab Smoke", {L"model-lab --objective conservative --max-symbols 3 --max-scan 20 --limit 500"}},
            };
        } else if (page_index_ == 1) {
            quick_actions_ = {
                {L"Live Paper Step", {L"live --paper --steps 1"}},
                {L"Paper Iteration", {L"autonomous start --paper --iterations 1"}},
                {L"Autonomous Status", {L"autonomous status"}},
                {L"Pause Bot", {L"autonomous pause"}},
                {L"Stop Bot", {L"autonomous stop"}},
                {L"Positions", {L"positions"}},
                {L"Reconcile", {L"reconcile"}},
                {L"Close Bot Positions", {L"close all"}},
            };
        } else if (page_index_ == 2) {
            quick_actions_ = {
                {L"Conservative Lab", {L"model-lab --objective conservative --max-symbols 3 --max-scan 20 --limit 500"}},
                {L"Regular Lab", {L"model-lab --objective regular --max-symbols 3 --max-scan 20 --limit 500 --market futures"}},
                {L"Aggressive Lab", {L"model-lab --objective aggressive --max-symbols 3 --max-scan 20 --limit 500 --market futures"}},
                {L"AI Review", {L"ai-review --report data/model_lab/model_lab_report.json"}},
                {L"Train Suite Help", {L"train-suite --help"}},
                {L"Backtest Panel", {L"backtest-panel --help"}},
                {L"Tune Help", {L"tune --help"}},
                {L"Objectives", {L"objectives"}},
            };
        } else if (page_index_ == 3) {
            quick_actions_ = {
                {L"Risk Paper", {L"risk --paper"}},
                {L"Universe Gate", {L"universe"}},
                {L"Reconcile", {L"reconcile"}},
                {L"Audit", {L"audit"}},
                {L"Doctor", {L"doctor"}},
                {L"Signals", {L"signals"}},
                {L"Source Grades", {L"source-grades"}},
                {L"Report", {L"report"}},
            };
        } else if (page_index_ == 4) {
            quick_actions_ = {
                {L"API Budget", {L"api-budget --compact"}},
                {L"Data Health", {L"data-health --interval 1s --market spot --json"}},
                {L"Archive Sync Help", {L"archive-sync --help"}},
                {L"Data Sync Help", {L"data-sync --help"}},
                {L"Fetch Help", {L"fetch --help"}},
                {L"Signal Sources", {L"source-grades"}},
            };
        } else if (page_index_ == 5) {
            quick_actions_ = {
                {L"AI Preflight", {L"ai"}},
                {L"Compute", {L"compute"}},
                {L"Configure", {L"configure --help"}},
                {L"Strategy", {L"strategy --help"}},
                {L"Menu", {L"menu --help"}},
                {L"Shell", {L"shell --help"}},
            };
        } else {
            quick_actions_ = {
                {L"Selected Help", {}},
                {L"Objectives", {L"objectives"}},
                {L"Doctor", {L"doctor"}},
                {L"Compute", {L"compute"}},
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
        }
        if (command.option_count > shown) {
            preview += L", ... +";
            preview += std::to_wstring(command.option_count - shown);
        }
        return preview;
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
        if (running_.exchange(true)) {
            append_output(L"\r\nA command is already running.\r\n");
            return;
        }
        EnableWindow(run_selected_, FALSE);
        SetWindowTextW(output_label_, L"Output - running");
        std::thread([this, commands = std::move(commands)] {
            for (const std::wstring& command : commands) {
                append_output(L"\r\n> simple-ai-trading " + command + L"\r\n");
                append_output(execute_cli(command));
            }
            running_ = false;
            refresh_api_budget_async(true);
            if (smoke_) {
                write_smoke_log();
            }
            PostMessageW(hwnd_, WM_APP + 1, 0, 0);
            if (smoke_) {
                PostMessageW(hwnd_, WM_CLOSE, 0, 0);
            }
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
            {
                std::lock_guard lock(api_budget_mutex_);
                api_budget_ = text;
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
        SetWindowTextW(status_bar_, snapshot.c_str());
    }

    std::wstring execute_cli(const std::wstring& args) {
        if (dry_run_enabled()) {
            return L"dry-run: simple-ai-trading " + args + L"\r\n\r\n(exit 0)\r\n";
        }
        std::wstring command = shell_command_for_cli(args);
        FILE* pipe = _wpopen(command.c_str(), L"r");
        std::wstring captured;
        if (!pipe) {
            return L"Failed to launch command.\r\n";
        }
        std::array<wchar_t, 1024> buffer{};
        while (fgetws(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
            captured += buffer.data();
        }
        int exit_code = _pclose(pipe);
        captured += L"\r\n(exit " + std::to_wstring(exit_code) + L")\r\n";
        return captured;
    }

    std::wstring execute_cli_first_line(const std::wstring& args) {
        if (dry_run_enabled()) {
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
