#pragma once

#include <windows.h>
#include <filesystem>

namespace simple_ai_trading::presentation {

// Run on the UI thread: its own GDI handles never cross a process boundary.
template <typename Render>
bool save_hidden_snapshot(HWND window, const std::filesystem::path& path, Render render) {
    RECT client{};
    if (!GetClientRect(window, &client) || IsWindowVisible(window)) return false;
    const LONG width = client.right, height = client.bottom;
    if (width < 640 || height < 480 || width > 8192 || height > 8192 ||
        static_cast<long long>(width) * height > 8388608) return false;
    BITMAPINFO info{};
    info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    info.bmiHeader.biWidth = width;
    info.bmiHeader.biHeight = -height;
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;
    const DWORD bytes = static_cast<DWORD>(width * height * 4);
    void* pixels = nullptr;
    HDC dc = CreateCompatibleDC(nullptr);
    HBITMAP bitmap = CreateDIBSection(dc, &info, DIB_RGB_COLORS, &pixels, nullptr, 0);
    if (!dc || !bitmap || !pixels) {
        if (bitmap) DeleteObject(bitmap);
        if (dc) DeleteDC(dc);
        return false;
    }
    HGDIOBJ previous = SelectObject(dc, bitmap);
    if (!previous || previous == HGDI_ERROR) {
        DeleteObject(bitmap);
        DeleteDC(dc);
        return false;
    }
    render(dc);
    struct Context { HWND window; HDC dc; } context{window, dc};
    EnumChildWindows(window, [](HWND child, LPARAM raw) -> BOOL {
        const auto* context = reinterpret_cast<Context*>(raw);
        if (GetParent(child) != context->window || !(GetWindowLongPtrW(child, GWL_STYLE) & WS_VISIBLE)) return TRUE;
        RECT rect{};
        GetWindowRect(child, &rect);
        MapWindowPoints(nullptr, context->window, reinterpret_cast<POINT*>(&rect), 2);
        const int saved = SaveDC(context->dc);
        SetViewportOrgEx(context->dc, rect.left, rect.top, nullptr);
        IntersectClipRect(context->dc, 0, 0, rect.right - rect.left, rect.bottom - rect.top);
        SendMessageW(child, WM_PRINT, reinterpret_cast<WPARAM>(context->dc), PRF_CLIENT | PRF_CHILDREN | PRF_ERASEBKGND);
        // Hidden dropdowns omit their selected owner-drawn field in WM_PRINT.
        // Use the native field bounds and the production WM_DRAWITEM handler.
        wchar_t class_name[32]{};
        GetClassNameW(child, class_name, 32);
        if (lstrcmpiW(class_name, L"ComboBox") == 0 &&
            (GetWindowLongPtrW(child, GWL_STYLE) & CBS_OWNERDRAWFIXED)) {
            COMBOBOXINFO combo{sizeof(COMBOBOXINFO)};
            RECT client{};
            if (GetComboBoxInfo(child, &combo) && GetClientRect(child, &client) &&
                combo.rcItem.left >= 0 && combo.rcItem.top >= 0 &&
                combo.rcItem.right <= client.right && combo.rcItem.bottom <= client.bottom) {
                DRAWITEMSTRUCT item{};
                item.CtlType = ODT_COMBOBOX;
                item.CtlID = static_cast<UINT>(GetDlgCtrlID(child));
                item.itemID = static_cast<UINT>(SendMessageW(child, CB_GETCURSEL, 0, 0));
                item.itemAction = ODA_DRAWENTIRE;
                item.itemState = ODS_COMBOBOXEDIT | (IsWindowEnabled(child) ? 0 : ODS_DISABLED);
                item.hwndItem = child;
                item.hDC = context->dc;
                item.rcItem = combo.rcItem;
                SendMessageW(context->window, WM_DRAWITEM, item.CtlID, reinterpret_cast<LPARAM>(&item));
            }
        }
        RestoreDC(context->dc, saved);
        return TRUE;
    }, reinterpret_cast<LPARAM>(&context));
    GdiFlush();
    BITMAPFILEHEADER header{};
    header.bfType = 0x4d42;
    header.bfOffBits = sizeof(header) + sizeof(BITMAPINFOHEADER);
    header.bfSize = header.bfOffBits + bytes;
    HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    bool success = false;
    if (file != INVALID_HANDLE_VALUE) {
        DWORD written = 0;
        success = WriteFile(file, &header, sizeof(header), &written, nullptr) && written == sizeof(header) &&
                  WriteFile(file, &info.bmiHeader, sizeof(BITMAPINFOHEADER), &written, nullptr) && written == sizeof(BITMAPINFOHEADER) &&
                  WriteFile(file, pixels, bytes, &written, nullptr) && written == bytes && FlushFileBuffers(file);
        if (!CloseHandle(file)) success = false;
    }
    SelectObject(dc, previous);
    DeleteObject(bitmap);
    DeleteDC(dc);
    return success;
}

}  // namespace simple_ai_trading::presentation
