// test_history_buffer.cpp ──────────────────────────────────────────────────

#include "test_harness.hpp"
#include "IOOptimizer.hpp"

int main() {
    std::cout << "=== HistoryBuffer Tests ===\n";

    TEST_BEGIN("basic_record_snapshot") {
        pec::HistoryBuffer hb(4);
        pec::Request r{};
        r.file_id = 10; hb.record(r);
        r.file_id = 20; hb.record(r);

        auto snap = hb.snapshot();
        ASSERT_EQ(snap.size(), 2u);
        ASSERT_EQ(snap[0].file_id, 10u);
        ASSERT_EQ(snap[1].file_id, 20u);
    } TEST_END()

    TEST_BEGIN("wrap_around") {
        pec::HistoryBuffer hb(3);
        pec::Request r{};
        for (uint64_t i = 1; i <= 5; ++i) {
            r.file_id = i;
            hb.record(r);
        }
        ASSERT_EQ(hb.size(), 3u);

        auto snap = hb.snapshot();
        ASSERT_EQ(snap.size(), 3u);
        // Should contain the last 3: 3, 4, 5 (oldest → newest)
        ASSERT_EQ(snap[0].file_id, 3u);
        ASSERT_EQ(snap[1].file_id, 4u);
        ASSERT_EQ(snap[2].file_id, 5u);
    } TEST_END()

    TEST_BEGIN("capacity_reports") {
        pec::HistoryBuffer hb(10);
        ASSERT_EQ(hb.capacity(), 10u);
        ASSERT_EQ(hb.size(), 0u);

        pec::Request r{};
        hb.record(r);
        ASSERT_EQ(hb.size(), 1u);
    } TEST_END()

    return test::test_main_return();
}
