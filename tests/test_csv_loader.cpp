// test_csv_loader.cpp ──────────────────────────────────────────────────────

#include "test_harness.hpp"
#include "DatasetLoader.hpp"

int main() {
    std::cout << "=== CsvTraceLoader Tests ===\n";

    TEST_BEGIN("parse_sample_trace") {
        pec::CsvTraceLoader loader("data/sample_trace.csv");
        int count = 0;
        while (auto r = loader.next()) {
            ++count;
            ASSERT_TRUE(r->request_id > 0);
            ASSERT_TRUE(r->user_id > 0);
            ASSERT_TRUE(r->file_id > 0);
            ASSERT_TRUE(r->size_bytes > 0);
            ASSERT_TRUE(r->arrival_ns > 0);
        }
        ASSERT_EQ(count, 10);
    } TEST_END()

    TEST_BEGIN("first_record_values") {
        pec::CsvTraceLoader loader("data/sample_trace.csv");
        auto r = loader.next();
        ASSERT_TRUE(r.has_value());
        ASSERT_EQ(r->request_id, 1u);
        ASSERT_EQ(r->user_id, 1u);
        ASSERT_EQ(r->file_id, 42u);
        ASSERT_EQ(r->size_bytes, 1048576u);
        ASSERT_EQ(r->arrival_ns, 1000000);
    } TEST_END()

    TEST_BEGIN("reset_replays") {
        pec::CsvTraceLoader loader("data/sample_trace.csv");
        while (loader.next()) {}  // exhaust
        loader.reset();
        auto r = loader.next();
        ASSERT_TRUE(r.has_value());
        ASSERT_EQ(r->request_id, 1u);
    } TEST_END()

    TEST_BEGIN("missing_file") {
        pec::CsvTraceLoader loader("data/nonexistent.csv");
        auto r = loader.next();
        ASSERT_FALSE(r.has_value());
    } TEST_END()

    return test::test_main_return();
}
