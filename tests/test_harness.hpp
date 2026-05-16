#pragma once
// test_harness.hpp ─────────────────────────────────────────────────────────
// Minimal test assertion macros.  Each test file has a standalone main()
// registered as a CTest target.  No external dependencies.
//
// Usage:
//   TEST_BEGIN("name") { ... assertions ... } TEST_END()

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace test {

inline int g_pass = 0;
inline int g_fail = 0;
inline std::string g_current_test;

inline void report_fail(const char* file, int line, const std::string& msg) {
    std::cerr << "  FAIL [" << file << ":" << line << "] " << msg << "\n";
    ++g_fail;
}

inline void report_pass() { ++g_pass; }

#define ASSERT_TRUE(expr)                                                     \
    do {                                                                       \
        if (!(expr)) {                                                         \
            std::ostringstream ss_;                                             \
            ss_ << #expr << " is false";                                       \
            test::report_fail(__FILE__, __LINE__, ss_.str());                  \
        } else {                                                               \
            test::report_pass();                                               \
        }                                                                      \
    } while (0)

#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))

#define ASSERT_EQ(a, b)                                                       \
    do {                                                                       \
        auto va_ = (a); auto vb_ = (b);                                        \
        if (va_ != vb_) {                                                      \
            std::ostringstream ss_;                                             \
            ss_ << #a << " == " << #b << " failed: " << va_ << " != " << vb_; \
            test::report_fail(__FILE__, __LINE__, ss_.str());                  \
        } else {                                                               \
            test::report_pass();                                               \
        }                                                                      \
    } while (0)

#define ASSERT_NE(a, b)                                                       \
    do {                                                                       \
        auto va_ = (a); auto vb_ = (b);                                        \
        if (va_ == vb_) {                                                      \
            std::ostringstream ss_;                                             \
            ss_ << #a << " != " << #b << " failed: both are " << va_;         \
            test::report_fail(__FILE__, __LINE__, ss_.str());                  \
        } else {                                                               \
            test::report_pass();                                               \
        }                                                                      \
    } while (0)

#define ASSERT_GE(a, b)                                                       \
    do {                                                                       \
        auto va_ = (a); auto vb_ = (b);                                        \
        if (va_ < vb_) {                                                       \
            std::ostringstream ss_;                                             \
            ss_ << #a << " >= " << #b << " failed: " << va_ << " < " << vb_; \
            test::report_fail(__FILE__, __LINE__, ss_.str());                  \
        } else {                                                               \
            test::report_pass();                                               \
        }                                                                      \
    } while (0)

#define ASSERT_LE(a, b)                                                       \
    do {                                                                       \
        auto va_ = (a); auto vb_ = (b);                                        \
        if (va_ > vb_) {                                                       \
            std::ostringstream ss_;                                             \
            ss_ << #a << " <= " << #b << " failed: " << va_ << " > " << vb_; \
            test::report_fail(__FILE__, __LINE__, ss_.str());                  \
        } else {                                                               \
            test::report_pass();                                               \
        }                                                                      \
    } while (0)

#define ASSERT_NEAR(a, b, tol)                                                \
    do {                                                                       \
        double va_ = (a); double vb_ = (b);                                    \
        if (std::fabs(va_ - vb_) > (tol)) {                                   \
            std::ostringstream ss_;                                             \
            ss_ << #a << " ≈ " << #b << " failed: " << va_ << " vs " << vb_  \
                << " (tol=" << (tol) << ")";                                   \
            test::report_fail(__FILE__, __LINE__, ss_.str());                  \
        } else {                                                               \
            test::report_pass();                                               \
        }                                                                      \
    } while (0)

#define TEST_BEGIN(name)                                                       \
    do {                                                                       \
        test::g_current_test = (name);                                         \
        std::cout << "  TEST " << test::g_current_test << " ...\n";

#define TEST_END()                                                             \
    } while (0);

inline int test_main_return() {
    std::cout << "\n  Results: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail ? EXIT_FAILURE : EXIT_SUCCESS;
}

}  // namespace test
