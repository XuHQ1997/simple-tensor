#ifndef UTILS_ALLOCATOR_H
#define UTILS_ALLOCATOR_H

#include <cstdlib>
#include <map>
#include <memory>
#include <functional>

#include "utils/base_config.h"

namespace st {

class Alloc {
public:
    class delete_handler {
    public:
        delete_handler(index_t size_) : size(size_) {}
        void operator()(void* ptr) { deallocate(ptr, size); }
    private:
        index_t size;
    };

    // I know it's weird here. The type has been already passed in as T, but the
    // function parameter still need the number of bytes, instead of objects.
    // And their relationship is 
    //          nbytes = nobjects * sizeof(T).
    // Check what I do in "tensor/storage.cpp", and you'll understand.
    // Or maybe changing the parameter here and doing some extra work in 
    // "tensor/storage.cpp" is better.
    template<typename T> 
    static std::shared_ptr<T> shared_allocate(index_t nbytes) {
        void* raw_ptr = allocate(nbytes);
        return std::shared_ptr<T>(
            static_cast<T*>(raw_ptr),
            delete_handler(nbytes)
        );
    }

    template<typename T>
    static std::unique_ptr<T, delete_handler> unique_allocate(index_t nbytes) {
        void* raw_ptr = allocate(nbytes);
        return std::unique_ptr<T, delete_handler>(
            static_cast<T*>(raw_ptr),
            delete_handler(nbytes)
        );
    }
private:
    Alloc() = default;
    ~Alloc() = default;
    static Alloc& self();
    static void* allocate(index_t size);
    static void deallocate(void* ptr, index_t size);

    struct free_deletor {
        void operator()(void* ptr) { std::free(ptr); }
    };
    std::multimap<index_t, std::unique_ptr<void, free_deletor>> cache_;
};

template std::unique_ptr<index_t, Alloc::delete_handler>
Alloc::unique_allocate(index_t bnytes);

} // namespace st
#endif