//
// Created by Kai Zhao on 12/9/19.
//

#ifndef SZ3_FILE_UTIL
#define SZ3_FILE_UTIL

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace SZ3 {

constexpr size_t FILE_IO_CHUNK_BYTES = static_cast<size_t>(1) << 28;

template <typename Type>
class MappedOutputFile {
public:
    MappedOutputFile() = default;

    MappedOutputFile(const char *file, size_t num_elements) { open(file, num_elements); }

    MappedOutputFile(const MappedOutputFile &) = delete;
    MappedOutputFile &operator=(const MappedOutputFile &) = delete;

    ~MappedOutputFile() { close(); }

    void open(const char *file, size_t num_elements) {
        close();
        count_ = num_elements;
        bytes_ = count_ * sizeof(Type);
        if (bytes_ == 0) {
            throw std::invalid_argument("mapped output file requires a non-empty buffer");
        }

#ifdef _WIN32
        file_handle_ = CreateFileA(
            file,
            GENERIC_READ | GENERIC_WRITE,
            0,
            nullptr,
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            nullptr);
        if (file_handle_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("failed to create output file for memory mapping");
        }

        LARGE_INTEGER size_value;
        size_value.QuadPart = static_cast<LONGLONG>(bytes_);
        if (!SetFilePointerEx(file_handle_, size_value, nullptr, FILE_BEGIN) || !SetEndOfFile(file_handle_)) {
            close();
            throw std::runtime_error("failed to resize mapped output file");
        }

        const DWORD size_high = static_cast<DWORD>((static_cast<uint64_t>(bytes_) >> 32u) & 0xffffffffu);
        const DWORD size_low = static_cast<DWORD>(static_cast<uint64_t>(bytes_) & 0xffffffffu);
        mapping_handle_ = CreateFileMappingA(file_handle_, nullptr, PAGE_READWRITE, size_high, size_low, nullptr);
        if (mapping_handle_ == nullptr) {
            close();
            throw std::runtime_error("failed to create file mapping");
        }

        data_ = static_cast<Type *>(MapViewOfFile(mapping_handle_, FILE_MAP_ALL_ACCESS, 0, 0, bytes_));
        if (data_ == nullptr) {
            close();
            throw std::runtime_error("failed to map output file view");
        }
#else
        fd_ = ::open(file, O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd_ < 0) {
            throw std::runtime_error("failed to create output file for memory mapping");
        }
        if (ftruncate(fd_, static_cast<off_t>(bytes_)) != 0) {
            close();
            throw std::runtime_error("failed to resize mapped output file");
        }
        void *mapped = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (mapped == MAP_FAILED) {
            data_ = nullptr;
            close();
            throw std::runtime_error("failed to map output file view");
        }
        data_ = static_cast<Type *>(mapped);
#endif
    }

    Type *data() const { return data_; }

    size_t size() const { return count_; }

    void flush() {
        if (data_ == nullptr) {
            return;
        }
#ifdef _WIN32
        FlushViewOfFile(data_, bytes_);
        if (file_handle_ != INVALID_HANDLE_VALUE) {
            FlushFileBuffers(file_handle_);
        }
#else
        msync(data_, bytes_, MS_SYNC);
#endif
    }

    void close() {
#ifdef _WIN32
        if (data_ != nullptr) {
            FlushViewOfFile(data_, bytes_);
            UnmapViewOfFile(data_);
            data_ = nullptr;
        }
        if (mapping_handle_ != nullptr) {
            CloseHandle(mapping_handle_);
            mapping_handle_ = nullptr;
        }
        if (file_handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
        }
#else
        if (data_ != nullptr) {
            msync(data_, bytes_, MS_SYNC);
            munmap(data_, bytes_);
            data_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
#endif
        count_ = 0;
        bytes_ = 0;
    }

private:
    Type *data_ = nullptr;
    size_t count_ = 0;
    size_t bytes_ = 0;
#ifdef _WIN32
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

/**
 * read binary file and put it to a existing memory space
 * @tparam Type
 * @param file
 * @param num
 * @param data
 */
template <typename Type>
void readfile(const char *file, const size_t num, Type *data) {
    std::ifstream fin(file, std::ios::binary);
    if (!fin) {
        std::cerr << " Error, Couldn't find the file: " << file << "\n";
        throw std::invalid_argument("Couldn't find the file");
    }
    fin.seekg(0, std::ios::end);
    if (fin.tellg() / sizeof(Type) != num) {
        throw std::invalid_argument("File size is not equal to the input setting");
    }
    fin.seekg(0, std::ios::beg);
    fin.read(reinterpret_cast<char *>(data), num * sizeof(Type));
    fin.close();
}

/**
 * read binary file and put it to a new memory space
 * @tparam Type
 * @param file
 * @param num
 * @return
 */
template <typename Type>
std::unique_ptr<Type[]> readfile(const char *file, size_t &num) {
    std::ifstream fin(file, std::ios::binary);
    if (!fin) {
        std::cerr << " Error, Couldn't find the file: " << file << std::endl;
        throw std::invalid_argument("Couldn't find the file");
    }
    fin.seekg(0, std::ios::end);
    num = fin.tellg() / sizeof(Type);
    fin.seekg(0, std::ios::beg);
    //        auto data = SZ3::compat::make_unique<Type[]>(num_elements);
    auto data = std::make_unique<Type[]>(num);
    fin.read(reinterpret_cast<char *>(&data[0]), num * sizeof(Type));
    fin.close();
    return data;
}

template <typename Type>
void writefile(const char *file, Type *data, size_t num_elements) {
    std::ofstream fout(file, std::ios::binary);
    if (!fout) {
        throw std::invalid_argument("Couldn't open the file for output");
    }
    const char *raw = reinterpret_cast<const char *>(&data[0]);
    size_t bytes_remaining = num_elements * sizeof(Type);
    while (bytes_remaining > 0) {
        const size_t chunk = std::min(bytes_remaining, FILE_IO_CHUNK_BYTES);
        fout.write(raw, static_cast<std::streamsize>(chunk));
        if (!fout) {
            throw std::invalid_argument("Couldn't write the file output");
        }
        raw += chunk;
        bytes_remaining -= chunk;
    }
    fout.close();
}

template <typename Type>
void writeTextFile(const char *file, Type *data, size_t num_elements) {
    std::ofstream fout(file);
    if (fout.is_open()) {
        for (size_t i = 0; i < num_elements; i++) {
            fout << data[i] << std::endl;
        }
        fout.close();
    } else {
        std::cerr << "Error, unable to open file for output: " << file << std::endl;
        throw std::invalid_argument("Couldn't open the file for output");
    }
}

}  // namespace SZ3

#endif
