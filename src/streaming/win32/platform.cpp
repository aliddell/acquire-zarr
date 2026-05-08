#include "definitions.hh"
#include "macros.hh"

#include <string_view>

#include <windows.h>

struct WinFileHandle
{
    HANDLE handle;
    DWORD sector_size; // only meaningful for aligned (NO_BUFFERING) handles
};

std::string
get_last_error_as_string()
{
    auto error_message_id = ::GetLastError();
    if (error_message_id == 0) {
        return std::string(); // No error message has been recorded
    }

    LPSTR message_buffer = nullptr;

    const auto format = FORMAT_MESSAGE_ALLOCATE_BUFFER |
                        FORMAT_MESSAGE_FROM_SYSTEM |
                        FORMAT_MESSAGE_IGNORE_INSERTS;
    const auto lang_id = MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT);
    size_t size = FormatMessageA(format,
                                 nullptr,
                                 error_message_id,
                                 lang_id,
                                 reinterpret_cast<LPSTR>(&message_buffer),
                                 0,
                                 nullptr);

    std::string message(message_buffer, size);

    LocalFree(message_buffer);

    return message;
}

void*
make_flags(bool aligned)
{
    auto* flags = new DWORD;
    *flags = aligned ? FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING
                     : FILE_FLAG_OVERLAPPED;
    return flags;
}

void
destroy_flags(const void* flags)
{
    delete static_cast<const DWORD*>(flags);
}

uint64_t
get_max_active_handles()
{
    return _getmaxstdio();
}

uint64_t
get_io_alignment()
{
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize;
}

void*
init_handle(const std::string& filename, const void* flags)
{
    const DWORD dw_flags = *static_cast<const DWORD*>(flags);

    auto* wfh = new WinFileHandle{ INVALID_HANDLE_VALUE, 0 };
    wfh->handle = CreateFileA(filename.c_str(),
                              GENERIC_WRITE,
                              FILE_SHARE_WRITE,
                              nullptr,
                              OPEN_ALWAYS,
                              dw_flags,
                              nullptr);

    if (wfh->handle == INVALID_HANDLE_VALUE) {
        const auto err = get_last_error_as_string();
        delete wfh;
        throw std::runtime_error("Failed to open file: '" +
                                 std::string(filename) + "': " + err);
    }

    if (dw_flags & FILE_FLAG_NO_BUFFERING) {
        FILE_STORAGE_INFO storage_info{};
        if (GetFileInformationByHandleEx(wfh->handle,
                                         FileStorageInfo,
                                         &storage_info,
                                         sizeof(storage_info))) {
            wfh->sector_size =
              storage_info.PhysicalBytesPerSectorForAtomicity;
        } else {
            wfh->sector_size = 4096; // safe fallback
        }
    }

    return wfh;
}

bool
seek_and_write(void* handle, size_t offset, ConstByteSpan data)
{
    CHECK(handle);
    const auto* wfh = static_cast<WinFileHandle*>(handle);

    auto* cur = reinterpret_cast<const char*>(data.data());
    auto* end = cur + data.size();

    int retries = 0;
    OVERLAPPED overlapped = {};
    overlapped.hEvent = CreateEventA(nullptr, TRUE, FALSE, nullptr);

    constexpr auto max_retries = 3;
    while (cur < end && retries < max_retries) {
        DWORD written = 0;
        const auto remaining = static_cast<DWORD>(end - cur); // may truncate
        overlapped.Pointer = reinterpret_cast<void*>(offset);
        if (!WriteFile(wfh->handle, cur, remaining, nullptr, &overlapped) &&
            GetLastError() != ERROR_IO_PENDING) {
            LOG_ERROR("Failed to write to file: ", get_last_error_as_string());
            CloseHandle(overlapped.hEvent);
            return false;
        }

        if (!GetOverlappedResult(wfh->handle, &overlapped, &written, TRUE)) {
            LOG_ERROR("Failed to get overlapped result: ",
                      get_last_error_as_string());
            CloseHandle(overlapped.hEvent);
            return false;
        }
        retries += written == 0 ? 1 : 0;
        offset += written;
        cur += written;
    }

    CloseHandle(overlapped.hEvent);
    return retries < max_retries;
}

bool
seek_and_write_aligned(void* handle, size_t offset, ConstByteSpan data)
{
    CHECK(handle);
    const auto* wfh = static_cast<WinFileHandle*>(handle);

    const size_t sector = wfh->sector_size;
    const size_t padded_size = (data.size() + sector - 1) & ~(sector - 1);

    // VirtualAlloc returns page-aligned, zeroed memory; padding needs no
    // explicit zero-fill
    auto* buf = static_cast<char*>(
      VirtualAlloc(nullptr, padded_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (!buf) {
        LOG_ERROR("Failed to allocate aligned write buffer: ",
                  get_last_error_as_string());
        return false;
    }
    memcpy(buf, data.data(), data.size());

    auto* cur = buf;
    auto* end = buf + padded_size;
    size_t cur_offset = offset;
    int retries = 0;
    OVERLAPPED overlapped = {};
    overlapped.hEvent = CreateEventA(nullptr, TRUE, FALSE, nullptr);

    constexpr auto max_retries = 3;
    while (cur < end && retries < max_retries) {
        DWORD written = 0;
        const auto remaining = static_cast<DWORD>(end - cur);
        overlapped.Pointer = reinterpret_cast<void*>(cur_offset);
        if (!WriteFile(wfh->handle, cur, remaining, nullptr, &overlapped) &&
            GetLastError() != ERROR_IO_PENDING) {
            LOG_ERROR("Failed to write to file: ", get_last_error_as_string());
            CloseHandle(overlapped.hEvent);
            VirtualFree(buf, 0, MEM_RELEASE);
            return false;
        }

        if (!GetOverlappedResult(wfh->handle, &overlapped, &written, TRUE)) {
            LOG_ERROR("Failed to get overlapped result: ",
                      get_last_error_as_string());
            CloseHandle(overlapped.hEvent);
            VirtualFree(buf, 0, MEM_RELEASE);
            return false;
        }
        retries += written == 0 ? 1 : 0;
        cur_offset += written;
        cur += written;
    }

    CloseHandle(overlapped.hEvent);
    VirtualFree(buf, 0, MEM_RELEASE);
    return retries < max_retries;
}

bool
flush_file(void* handle)
{
    CHECK(handle);
    if (const auto* wfh = static_cast<WinFileHandle*>(handle);
        wfh->handle != INVALID_HANDLE_VALUE) {
        return FlushFileBuffers(wfh->handle);
    }
    return true;
}

void
destroy_handle(void* handle)
{
    if (auto* wfh = static_cast<WinFileHandle*>(handle)) {
        if (wfh->handle != INVALID_HANDLE_VALUE) {
            FlushFileBuffers(wfh->handle);
            CloseHandle(wfh->handle);
        }
        delete wfh;
    }
}