// test_json_parser.cpp ─────────────────────────────────────────────────────

#include "test_harness.hpp"
#include "Json.hpp"

#include <stdexcept>

int main() {
    std::cout << "=== JSON Parser Tests ===\n";

    TEST_BEGIN("parse_simple_object") {
        auto v = pec::JsonValue::parse(R"({"a": 1, "b": "hello"})");
        ASSERT_TRUE(v.is_object());
        ASSERT_NEAR(v.get_or<double>("a", 0.0), 1.0, 0.001);
        ASSERT_EQ(v.get_or<std::string>("b", ""), std::string("hello"));
    } TEST_END()

    TEST_BEGIN("nested_dotted_path") {
        auto v = pec::JsonValue::parse(R"({"hw": {"queue": 2048}})");
        ASSERT_EQ(v.get_or<size_t>("hw.queue", 0u), 2048u);
    } TEST_END()

    TEST_BEGIN("missing_key_returns_default") {
        auto v = pec::JsonValue::parse(R"({"a": 1})");
        ASSERT_EQ(v.get_or<int>("missing", 42), 42);
        ASSERT_EQ(v.get_or<std::string>("nope", "fallback"), std::string("fallback"));
    } TEST_END()

    TEST_BEGIN("booleans") {
        auto v = pec::JsonValue::parse(R"({"t": true, "f": false})");
        ASSERT_TRUE(v.get_or<bool>("t", false));
        ASSERT_FALSE(v.get_or<bool>("f", true));
    } TEST_END()

    TEST_BEGIN("arrays") {
        auto v = pec::JsonValue::parse(R"({"arr": [1, 2, 3]})");
        auto* arr = v.find("arr");
        ASSERT_TRUE(arr != nullptr);
        ASSERT_TRUE(arr->is_array());
        ASSERT_EQ(arr->as_array().size(), 3u);
    } TEST_END()

    TEST_BEGIN("parse_error_throws") {
        bool caught = false;
        try {
            pec::JsonValue::parse(R"({bad json)");
        } catch (const std::runtime_error& e) {
            caught = true;
            std::string msg = e.what();
            ASSERT_TRUE(msg.find("line") != std::string::npos);
        }
        ASSERT_TRUE(caught);
    } TEST_END()

    TEST_BEGIN("escape_sequences") {
        auto v = pec::JsonValue::parse(R"({"s": "hello\nworld"})");
        auto s = v.get_or<std::string>("s", "");
        ASSERT_TRUE(s.find('\n') != std::string::npos);
    } TEST_END()

    return test::test_main_return();
}
