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

constexpr int kInitialWidth = 1480;
constexpr int kInitialHeight = 920;
constexpr int kMinWidth = 1120;
constexpr int kMinHeight = 720;
constexpr COLORREF kBg = RGB(18, 22, 25);
constexpr COLORREF kShell = RGB(24, 30, 34);
constexpr COLORREF kPanel = RGB(31, 38, 43);
constexpr COLORREF kPanel2 = RGB(40, 49, 55);
constexpr COLORREF kAccent = RGB(69, 168, 151);
constexpr COLORREF kDanger = RGB(139, 55, 60);
constexpr COLORREF kText = RGB(238, 242, 244);
constexpr COLORREF kMuted = RGB(172, 183, 190);
constexpr COLORREF kSubtle = RGB(115, 127, 135);

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
        int width = std::min(kInitialWidth, std::max(kMinWidth, work_width - 80));
        int height = std::min(kInitialHeight, std::max(kMinHeight, work_height - 80));
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
    HWND page_list_{};
    HWND command_label_{};
    HWND command_combo_{};
    HWND args_label_{};
    HWND args_edit_{};
    HWND help_label_{};
    HWND output_label_{};
    HWND output_edit_{};
    HWND run_selected_{};
    HWND selected_help_{};
    HWND stop_all_{};
    HWND ai_preflight_{};
    HWND risk_report_{};
    HWND model_lab_{};
    HWND backtest_chart_{};
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
    std::mutex output_mutex_;
    std::atomic_bool running_{false};
    bool smoke_ = false;

    static constexpr std::array<const wchar_t*, 6> kPages{
        L"Operate",
        L"Research",
        L"Risk",
        L"Data",
        L"Settings",
        L"CLI Parity",
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
        case WM_CTLCOLORSTATIC:
        case WM_CTLCOLOREDIT:
        case WM_CTLCOLORLISTBOX:
        case WM_CTLCOLORBTN:
            return color_control(reinterpret_cast<HDC>(wparam), reinterpret_cast<HWND>(lparam), message);
        case WM_PAINT:
            paint();
            return 0;
        case WM_APP + 1:
            sync_output();
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
        rebuild_fonts();
        create_controls();
        populate_pages();
        refresh_page();
        sync_output();
        layout();
    }

    void cleanup() {
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

    HFONT make_font(int points, int weight = FW_NORMAL, const wchar_t* face = L"Segoe UI") const {
        return CreateFontW(
            -MulDiv(points, dpi_, 72),
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
        body_font_ = make_font(12, FW_NORMAL);
        small_font_ = make_font(10, FW_NORMAL);
        mono_font_ = make_font(11, FW_NORMAL, L"Consolas");
        for (HWND control : all_controls()) {
            if (control) {
                SendMessageW(control, WM_SETFONT, reinterpret_cast<WPARAM>(body_font_), TRUE);
            }
        }
        if (title_) SendMessageW(title_, WM_SETFONT, reinterpret_cast<WPARAM>(title_font_), TRUE);
        if (subtitle_) SendMessageW(subtitle_, WM_SETFONT, reinterpret_cast<WPARAM>(small_font_), TRUE);
        if (safety_) SendMessageW(safety_, WM_SETFONT, reinterpret_cast<WPARAM>(small_font_), TRUE);
        if (output_edit_) SendMessageW(output_edit_, WM_SETFONT, reinterpret_cast<WPARAM>(mono_font_), TRUE);
    }

    std::vector<HWND> all_controls() const {
        std::vector<HWND> controls{
            title_,       subtitle_,      safety_,       page_list_,     command_label_,
            command_combo_, args_label_, args_edit_,    help_label_,    output_label_,
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
        subtitle_ = create_control(L"STATIC", L"Autonomous multi-asset day-trading workstation", SS_LEFT, 0);
        safety_ = create_control(
            L"STATIC",
            L"Default: conservative, testnet, no leverage, no profit reinvestment. Stop also closes local positions.",
            SS_LEFT | SS_NOPREFIX,
            0);
        page_list_ = create_control(L"LISTBOX", L"", LBS_NOTIFY | WS_TABSTOP | WS_VSCROLL, kPageListId, WS_EX_CLIENTEDGE);
        command_label_ = create_control(L"STATIC", L"Command", SS_LEFT, 0);
        command_combo_ = create_control(
            L"COMBOBOX",
            L"",
            CBS_DROPDOWNLIST | CBS_HASSTRINGS | WS_TABSTOP | WS_VSCROLL,
            kCommandComboId,
            WS_EX_CLIENTEDGE);
        args_label_ = create_control(L"STATIC", L"Extra Arguments", SS_LEFT, 0);
        args_edit_ = create_control(L"EDIT", L"", ES_AUTOHSCROLL | WS_TABSTOP, kArgsEditId, WS_EX_CLIENTEDGE);
        help_label_ = create_control(L"STATIC", L"", SS_LEFT | SS_NOPREFIX, 0);
        output_label_ = create_control(L"STATIC", L"Output", SS_LEFT, 0);
        output_edit_ = create_control(
            L"EDIT",
            L"",
            ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL | WS_VSCROLL | WS_TABSTOP,
            kOutputEditId,
            WS_EX_CLIENTEDGE);
        run_selected_ = create_control(L"BUTTON", L"Run Selected", BS_PUSHBUTTON | WS_TABSTOP, kRunSelectedId);
        selected_help_ = create_control(L"BUTTON", L"Selected Help", BS_PUSHBUTTON | WS_TABSTOP, kSelectedHelpId);
        stop_all_ = create_control(L"BUTTON", L"Stop And Close All", BS_PUSHBUTTON | WS_TABSTOP, kStopAllId);
        ai_preflight_ = create_control(L"BUTTON", L"AI Preflight", BS_PUSHBUTTON | WS_TABSTOP, kAiPreflightId);
        risk_report_ = create_control(L"BUTTON", L"Risk Report", BS_PUSHBUTTON | WS_TABSTOP, kRiskReportId);
        model_lab_ = create_control(L"BUTTON", L"Model Lab", BS_PUSHBUTTON | WS_TABSTOP, kModelLabId);
        backtest_chart_ = create_control(L"BUTTON", L"Backtest Chart", BS_PUSHBUTTON | WS_TABSTOP, kBacktestChartId);
        for (int i = 0; i < static_cast<int>(quick_buttons_.size()); ++i) {
            quick_buttons_[static_cast<std::size_t>(i)] =
                create_control(L"BUTTON", L"", BS_PUSHBUTTON | WS_TABSTOP, kQuickBaseId + i);
        }
    }

    void populate_pages() {
        SendMessageW(page_list_, LB_RESETCONTENT, 0, 0);
        for (const wchar_t* page : kPages) {
            SendMessageW(page_list_, LB_ADDSTRING, 0, reinterpret_cast<LPARAM>(page));
        }
        SendMessageW(page_list_, LB_SETCURSEL, 0, 0);
    }

    void layout() {
        if (!hwnd_ || !title_) return;
        RECT client{};
        GetClientRect(hwnd_, &client);
        const int pad = scale(22);
        const int top = scale(82);
        const int rail = scale(232);
        const int gap = scale(18);
        const int right = client.right - pad;
        const int bottom = client.bottom - pad;
        const int main_left = pad + rail + gap;
        const int main_width = std::max(scale(640), right - main_left);

        MoveWindow(title_, pad, scale(12), scale(360), scale(38), TRUE);
        MoveWindow(subtitle_, pad, scale(50), scale(620), scale(24), TRUE);
        MoveWindow(safety_, main_left, scale(18), main_width, scale(46), TRUE);
        MoveWindow(page_list_, pad, top, rail, bottom - top, TRUE);

        const int row_top = top;
        const int combo_w = std::max(scale(300), main_width / 3);
        const int args_left = main_left + combo_w + gap;
        const int run_w = scale(138);
        MoveWindow(command_label_, main_left, row_top, combo_w, scale(22), TRUE);
        MoveWindow(command_combo_, main_left, row_top + scale(25), combo_w, scale(260), TRUE);
        MoveWindow(args_label_, args_left, row_top, std::max(scale(240), right - args_left - run_w - gap), scale(22), TRUE);
        MoveWindow(args_edit_, args_left, row_top + scale(25), std::max(scale(240), right - args_left - run_w - gap), scale(34), TRUE);
        MoveWindow(run_selected_, right - run_w, row_top + scale(24), run_w, scale(36), TRUE);
        MoveWindow(selected_help_, right - run_w, row_top + scale(66), run_w, scale(34), TRUE);

        const int help_top = row_top + scale(72);
        MoveWindow(help_label_, main_left, help_top, std::max(scale(420), main_width - run_w - gap), scale(56), TRUE);

        const int quick_top = top + scale(148);
        const int quick_cols = 4;
        const int quick_gap = scale(12);
        const int quick_h = scale(40);
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

        const int tools_top = quick_top + scale(168);
        const int tool_gap = scale(10);
        const int tool_w = (main_width - (tool_gap * 4)) / 5;
        MoveWindow(stop_all_, main_left, tools_top, tool_w, scale(38), TRUE);
        MoveWindow(ai_preflight_, main_left + (tool_w + tool_gap), tools_top, tool_w, scale(38), TRUE);
        MoveWindow(risk_report_, main_left + (2 * (tool_w + tool_gap)), tools_top, tool_w, scale(38), TRUE);
        MoveWindow(model_lab_, main_left + (3 * (tool_w + tool_gap)), tools_top, tool_w, scale(38), TRUE);
        MoveWindow(backtest_chart_, main_left + (4 * (tool_w + tool_gap)), tools_top, tool_w, scale(38), TRUE);

        const int output_top = tools_top + scale(58);
        MoveWindow(output_label_, main_left, output_top - scale(28), main_width, scale(22), TRUE);
        MoveWindow(output_edit_, main_left, output_top, main_width, std::max(scale(180), bottom - output_top), TRUE);
    }

    void paint() {
        PAINTSTRUCT ps{};
        HDC dc = BeginPaint(hwnd_, &ps);
        RECT client{};
        GetClientRect(hwnd_, &client);
        HBRUSH bg = CreateSolidBrush(kBg);
        FillRect(dc, &client, bg);
        DeleteObject(bg);

        const int pad = scale(22);
        const int rail = scale(232);
        const int top = scale(82);
        RECT nav_panel{pad - scale(8), top - scale(8), pad + rail + scale(8), client.bottom - pad + scale(8)};
        fill_rect(dc, nav_panel, kPanel);
        RECT top_panel{pad + rail + scale(10), scale(10), client.right - pad + scale(8), top - scale(12)};
        fill_rect(dc, top_panel, kShell);
        EndPaint(hwnd_, &ps);
    }

    void fill_rect(HDC dc, const RECT& rect, COLORREF color) {
        HBRUSH brush = CreateSolidBrush(color);
        FillRect(dc, &rect, brush);
        DeleteObject(brush);
    }

    LRESULT color_control(HDC dc, HWND control, UINT message) {
        SetTextColor(dc, kText);
        SetBkColor(dc, kBg);
        if (control == output_edit_) {
            SetTextColor(dc, RGB(218, 228, 232));
            SetBkColor(dc, RGB(14, 18, 21));
            return reinterpret_cast<LRESULT>(edit_brush_);
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
        refresh_command_combo();
        refresh_quick_actions();
        update_selected_help();
    }

    void refresh_command_combo() {
        command_entries_.clear();
        SendMessageW(command_combo_, CB_RESETCONTENT, 0, 0);
        if (page_index_ == 0) {
            add_group(L"Operate", {L"status", L"connect", L"compute", L"live", L"autonomous", L"positions", L"close"});
        } else if (page_index_ == 1) {
            add_group(L"Research", {L"model-lab", L"train-suite", L"train", L"prepare", L"tune", L"backtest", L"backtest-chart", L"backtest-panel", L"evaluate", L"objectives"});
        } else if (page_index_ == 2) {
            add_group(L"Risk", {L"risk", L"audit", L"doctor", L"universe", L"signals", L"signals-benchmark", L"source-grades", L"report"});
        } else if (page_index_ == 3) {
            add_group(L"Data", {L"data-sync", L"fetch", L"configure", L"strategy", L"spot-roundtrip"});
        } else if (page_index_ == 4) {
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
                {L"Status", {L"status"}},
                {L"Connect", {L"connect"}},
                {L"Compute GPU", {L"compute"}},
                {L"Live Paper Step", {L"live --paper --steps 1"}},
                {L"Autonomous Status", {L"autonomous status"}},
                {L"Paper Iteration", {L"autonomous start --paper --iterations 1"}},
                {L"Pause", {L"autonomous pause"}},
                {L"Stop", {L"autonomous stop"}},
                {L"Positions", {L"positions"}},
                {L"Close All", {L"close all"}},
            };
        } else if (page_index_ == 1) {
            quick_actions_ = {
                {L"Model Lab Conservative", {L"model-lab --objective conservative --max-symbols 3 --max-scan 20 --limit 500"}},
                {L"Model Lab Regular", {L"model-lab --objective regular --max-symbols 3 --max-scan 20 --limit 500 --market futures"}},
                {L"Train Suite", {L"train-suite --help"}},
                {L"Prepare Pipeline", {L"prepare --help"}},
                {L"Backtest", {L"backtest --help"}},
                {L"Backtest Chart", {L"backtest-chart"}},
                {L"Backtest Panel", {L"backtest-panel --help"}},
                {L"Tune", {L"tune --help"}},
                {L"Objectives", {L"objectives"}},
            };
        } else if (page_index_ == 2) {
            quick_actions_ = {
                {L"Risk Paper", {L"risk --paper"}},
                {L"Audit", {L"audit"}},
                {L"Doctor", {L"doctor"}},
                {L"Universe", {L"universe"}},
                {L"Signals", {L"signals"}},
                {L"Signals Benchmark", {L"signals-benchmark --help"}},
                {L"Source Grades", {L"source-grades"}},
                {L"Report", {L"report"}},
            };
        } else if (page_index_ == 3) {
            quick_actions_ = {
                {L"Data Sync", {L"data-sync --help"}},
                {L"Fetch", {L"fetch --help"}},
                {L"Configure", {L"configure --help"}},
                {L"Strategy", {L"strategy --help"}},
                {L"Spot Roundtrip", {L"spot-roundtrip --help"}},
            };
        } else if (page_index_ == 4) {
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
        SetWindowTextW(output_label_, running_ ? L"Output - running" : L"Output");
    }

    std::wstring execute_cli(const std::wstring& args) {
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

    static bool env_present(const wchar_t* name) {
        std::array<wchar_t, 8> value{};
        DWORD size = GetEnvironmentVariableW(name, value.data(), static_cast<DWORD>(value.size()));
        return size > 0;
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
            std::filesystem::path venv = root / L".venv311" / L"Scripts" / L"python.exe";
            if (std::filesystem::exists(venv)) {
                return cmd_quote(venv.wstring());
            }
        }
        return L"py -3.11";
    }

    static std::wstring shell_command_for_cli(const std::wstring& args) {
        std::filesystem::path root = repo_root();
        std::wstring command = L"cmd.exe /d /s /c \"";
        if (!root.empty()) {
            std::wstring root_text = root.wstring();
            command += L"cd /d " + cmd_quote(root_text) + L" && ";
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
