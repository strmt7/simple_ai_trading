#include "command_contract.hpp"

#include <dwmapi.h>
#include <shellapi.h>
#include <windows.h>
#include <windowsx.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace app {
using simple_ai_trading::native_contract::kCommandCount;
using simple_ai_trading::native_contract::kCommands;

constexpr int kWidth = 1180;
constexpr int kHeight = 760;
constexpr COLORREF kBg = RGB(21, 24, 27);
constexpr COLORREF kPanel = RGB(31, 36, 40);
constexpr COLORREF kPanel2 = RGB(38, 45, 50);
constexpr COLORREF kAccent = RGB(77, 166, 156);
constexpr COLORREF kWarn = RGB(235, 187, 83);
constexpr COLORREF kText = RGB(235, 238, 241);
constexpr COLORREF kMuted = RGB(154, 164, 172);

struct Button {
    RECT rect{};
    const wchar_t* label{};
    int action{};
};

class MainWindow {
  public:
    int run(HINSTANCE instance, int show) {
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
        RECT frame{0, 0, kWidth, kHeight};
        if (!AdjustWindowRectExForDpi(&frame, WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX, FALSE, 0, 96)) {
            AdjustWindowRect(&frame, WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX, FALSE);
        }
        hwnd_ = CreateWindowExW(0, wc.lpszClassName, L"Simple AI Trading", WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
                                CW_USEDEFAULT, CW_USEDEFAULT, frame.right - frame.left, frame.bottom - frame.top,
                                nullptr, nullptr, instance, this);
        if (!hwnd_) {
            log_startup_failure(L"CreateWindowExW");
            return 1;
        }
        BOOL dark = TRUE;
        COLORREF caption = RGB(27, 30, 33);
        COLORREF caption_text = kText;
        COLORREF border = RGB(54, 72, 78);
        DwmSetWindowAttribute(hwnd_, DWMWA_USE_IMMERSIVE_DARK_MODE, &dark, sizeof(dark));
        DwmSetWindowAttribute(hwnd_, DWMWA_CAPTION_COLOR, &caption, sizeof(caption));
        DwmSetWindowAttribute(hwnd_, DWMWA_TEXT_COLOR, &caption_text, sizeof(caption_text));
        DwmSetWindowAttribute(hwnd_, DWMWA_BORDER_COLOR, &border, sizeof(border));
        ShowWindow(hwnd_, show);
        UpdateWindow(hwnd_);
        run_command(L"compute");
        MSG msg{};
        while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        return static_cast<int>(msg.wParam);
    }

  private:
    HWND hwnd_{};
    int selected_{0};
    std::wstring output_{L"Native Win32 command surface. The command list is generated from the Python CLI parser.\r\n"};
    std::atomic_bool running_{false};
    std::array<Button, 6> buttons_{};

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
        case WM_PAINT:
            paint();
            return 0;
        case WM_LBUTTONDOWN:
            click(GET_X_LPARAM(lparam), GET_Y_LPARAM(lparam));
            return 0;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        case WM_APP + 1:
            InvalidateRect(hwnd_, nullptr, FALSE);
            return 0;
        default:
            return DefWindowProcW(hwnd_, message, wparam, lparam);
        }
    }

    HFONT make_font(int size, int weight = FW_NORMAL) {
        return CreateFontW(size, 0, 0, 0, weight, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                           CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Segoe UI");
    }

    void fill(HDC dc, RECT rect, COLORREF color) {
        HBRUSH brush = CreateSolidBrush(color);
        FillRect(dc, &rect, brush);
        DeleteObject(brush);
    }

    void text(HDC dc, const std::wstring& value, RECT rect, COLORREF color, HFONT font, UINT format) {
        SelectObject(dc, font);
        SetBkMode(dc, TRANSPARENT);
        SetTextColor(dc, color);
        DrawTextW(dc, value.c_str(), static_cast<int>(value.size()), &rect, format);
    }

    void paint() {
        PAINTSTRUCT ps{};
        HDC dc = BeginPaint(hwnd_, &ps);
        RECT client{};
        GetClientRect(hwnd_, &client);
        HDC mem = CreateCompatibleDC(dc);
        HBITMAP bitmap = CreateCompatibleBitmap(dc, client.right, client.bottom);
        auto* old = SelectObject(mem, bitmap);
        fill(mem, client, kBg);
        HFONT title = make_font(30, FW_SEMIBOLD);
        HFONT body = make_font(17);
        HFONT small_font = make_font(14);
        HFONT mono = CreateFontW(15, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                                 CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, FIXED_PITCH | FF_MODERN, L"Consolas");

        RECT nav{18, 18, 270, 724};
        fill(mem, nav, kPanel);
        RECT title_rect{296, 20, 1140, 58};
        text(mem, L"Simple AI Trading", title_rect, kText, title, DT_LEFT | DT_SINGLELINE);
        RECT subtitle{296, 58, 1140, 92};
        text(mem, L"Native Win32 operator console with generated CLI parity, model-lab workflows, DirectML checks, and stop controls.",
             subtitle, kMuted, body, DT_LEFT | DT_WORDBREAK);

        RECT nav_title{34, 34, 250, 58};
        text(mem, L"Workflows", nav_title, kText, body, DT_LEFT | DT_SINGLELINE);
        int y = 70;
        for (int i = 0; i < kCommandCount && y < 690; ++i) {
            RECT row{30, y, 258, y + 24};
            if (i == selected_) {
                fill(mem, row, kPanel2);
                RECT strip{30, y, 35, y + 24};
                fill(mem, strip, kAccent);
            }
            text(mem, kCommands[i].name, RECT{42, y + 3, 254, y + 24}, i == selected_ ? kText : kMuted, small_font,
                 DT_LEFT | DT_SINGLELINE | DT_END_ELLIPSIS);
            y += 26;
        }

        draw_cards(mem, body, small_font);
        draw_buttons(mem, body);
        RECT out_panel{296, 246, 1140, 724};
        fill(mem, out_panel, RGB(18, 21, 24));
        RECT out_title{314, 262, 1120, 286};
        text(mem, running_ ? L"Command output - running" : L"Command output", out_title, running_ ? kWarn : kText, body,
             DT_LEFT | DT_SINGLELINE);
        RECT out_text{314, 294, 1120, 708};
        text(mem, tail(output_, 9000), out_text, RGB(210, 218, 224), mono, DT_LEFT | DT_TOP | DT_WORDBREAK);

        BitBlt(dc, 0, 0, client.right, client.bottom, mem, 0, 0, SRCCOPY);
        SelectObject(mem, old);
        DeleteObject(bitmap);
        DeleteDC(mem);
        DeleteObject(title);
        DeleteObject(body);
        DeleteObject(small_font);
        DeleteObject(mono);
        EndPaint(hwnd_, &ps);
    }

    void draw_cards(HDC dc, HFONT body, HFONT small_font) {
        RECT command{296, 102, 730, 218};
        RECT safety{748, 102, 1140, 218};
        fill(dc, command, kPanel);
        fill(dc, safety, kPanel);
        std::wstring name = kCommandCount > 0 ? kCommands[selected_].name : L"";
        std::wstring help = kCommandCount > 0 ? kCommands[selected_].help : L"No commands generated";
        text(dc, L"Selected Command", RECT{314, 118, 700, 142}, kMuted, small_font, DT_LEFT | DT_SINGLELINE);
        text(dc, name, RECT{314, 144, 700, 172}, kText, body, DT_LEFT | DT_SINGLELINE);
        text(dc, help, RECT{314, 174, 704, 210}, kMuted, small_font, DT_LEFT | DT_WORDBREAK | DT_END_ELLIPSIS);
        text(dc, L"Safety Gates", RECT{766, 118, 1110, 142}, kMuted, small_font, DT_LEFT | DT_SINGLELINE);
        text(dc, L"Stop closes local autonomous positions. Model Lab rejects non-profitable candidates. CPU mode disables AI.", RECT{766, 146, 1112, 206}, kText, small_font,
             DT_LEFT | DT_WORDBREAK);
    }

    void draw_buttons(HDC dc, HFONT font) {
        buttons_ = {{
            {RECT{296, 224, 410, 238}, L"Run Selected", 1},
            {RECT{422, 224, 506, 238}, L"AI", 2},
            {RECT{518, 224, 612, 238}, L"Risk", 3},
            {RECT{624, 224, 744, 238}, L"Model Lab", 4},
            {RECT{756, 224, 874, 238}, L"Stop", 5},
            {RECT{886, 224, 1016, 238}, L"Backtest Chart", 6},
        }};
        for (const auto& button : buttons_) {
            RECT box{button.rect.left, button.rect.top - 2, button.rect.right, button.rect.bottom + 18};
            fill(dc, box, button.action == 5 ? RGB(93, 45, 49) : kPanel2);
            text(dc, button.label, RECT{box.left + 10, box.top + 6, box.right - 8, box.bottom}, kText, font,
                 DT_LEFT | DT_SINGLELINE);
        }
    }

    static std::wstring tail(const std::wstring& value, std::size_t max_chars) {
        if (value.size() <= max_chars) {
            return value;
        }
        return L"...\r\n" + value.substr(value.size() - max_chars);
    }

    static std::wstring cli_prefix() {
        std::array<wchar_t, 32768> python{};
        DWORD size = GetEnvironmentVariableW(L"SIMPLE_AI_TRADING_PYTHON", python.data(), static_cast<DWORD>(python.size()));
        if (size > 0 && size < python.size()) {
            std::wstring path = python.data();
            std::replace(path.begin(), path.end(), L'"', L' ');
            return L"\"" + path + L"\" -m simple_ai_trading";
        }
        return L"py -3.11 -m simple_ai_trading";
    }

    void click(int x, int y) {
        if (x >= 30 && x <= 258 && y >= 70) {
            int index = (y - 70) / 26;
            if (index >= 0 && index < kCommandCount) {
                selected_ = index;
                InvalidateRect(hwnd_, nullptr, FALSE);
                return;
            }
        }
        POINT point{x, y};
        for (const auto& button : buttons_) {
            RECT hot{button.rect.left, button.rect.top - 2, button.rect.right, button.rect.bottom + 18};
            if (PtInRect(&hot, point)) {
                handle_action(button.action);
                return;
            }
        }
    }

    void handle_action(int action) {
        switch (action) {
        case 1:
            if (kCommandCount > 0) run_command(kCommands[selected_].name);
            break;
        case 2:
            run_command(L"ai");
            break;
        case 3:
            run_command(L"risk --paper");
            break;
        case 4:
            run_command(L"model-lab --max-symbols 3 --limit 500");
            break;
        case 5:
            run_command(L"autonomous stop");
            break;
        case 6:
            run_command(L"backtest-chart");
            break;
        default:
            break;
        }
    }

    void run_command(const std::wstring& args) {
        if (running_) return;
        running_ = true;
        output_ += L"\r\n> simple-ai-trading " + args + L"\r\n";
        InvalidateRect(hwnd_, nullptr, FALSE);
        std::thread([this, args] {
            std::wstring command = L"cmd.exe /c \"" + cli_prefix() + L" " + args + L" 2>&1\"";
            FILE* pipe = _wpopen(command.c_str(), L"r");
            std::wstring captured;
            if (!pipe) {
                captured = L"Failed to launch command.\r\n";
            } else {
                std::array<wchar_t, 512> buffer{};
                while (fgetws(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
                    captured += buffer.data();
                }
                int exit_code = _pclose(pipe);
                captured += L"\r\n(exit " + std::to_wstring(exit_code) + L")\r\n";
            }
            output_ += captured;
            running_ = false;
            PostMessageW(hwnd_, WM_APP + 1, 0, 0);
        }).detach();
    }
};
} // namespace app

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int show) {
    app::MainWindow window;
    return window.run(instance, show);
}
