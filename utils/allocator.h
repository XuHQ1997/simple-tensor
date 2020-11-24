#include <cstdlib>
#include <map>
#include <memory>

#include "utils/base_config.h"

namespace st {

class Alloc {
public:
    template<typename T>
    static T* allocate(index_t size) {
        return static_cast<T*>(self().__allocate(size));
    }

    template<typename T> 
    static void deallocate(T* ptr, index_t size) {
        self().__deallocate(static_cast<void*>(ptr), size);
    }

private:
    Alloc() = default;
    ~Alloc() = default;
    static Alloc& self();
    void* __allocate(index_t size);
    void __deallocate(void* ptr, index_t size);

    unsigned int round_up(int x);

    struct free_deletor {
        void operator()(void* ptr) { std::free(ptr); }
    };
    std::multimap<index_t, std::unique_ptr<void, free_deletor>> cache;
};

} // namespace st