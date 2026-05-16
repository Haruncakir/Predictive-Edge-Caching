// test_synthetic_engine.cpp ────────────────────────────────────────────────

#include "test_harness.hpp"
#include "TrafficEngine.hpp"

#include <map>

int main() {
    std::cout << "=== SyntheticEngine Tests ===\n";

    TEST_BEGIN("deterministic_with_same_seed") {
        pec::SyntheticConfig cfg;
        cfg.catalog_size = 100;
        cfg.seed = 42;

        pec::SyntheticEngine eng1(cfg);
        pec::SyntheticEngine eng2(cfg);

        for (int i = 0; i < 100; ++i) {
            auto r1 = eng1.generate();
            auto r2 = eng2.generate();
            ASSERT_TRUE(r1.has_value());
            ASSERT_TRUE(r2.has_value());
            ASSERT_EQ(r1->file_id, r2->file_id);
            ASSERT_EQ(r1->user_id, r2->user_id);
            ASSERT_EQ(r1->arrival_ns, r2->arrival_ns);
        }
    } TEST_END()

    TEST_BEGIN("file_ids_in_range") {
        pec::SyntheticConfig cfg;
        cfg.catalog_size = 50;

        pec::SyntheticEngine eng(cfg);
        for (int i = 0; i < 1000; ++i) {
            auto r = eng.generate();
            ASSERT_TRUE(r.has_value());
            ASSERT_GE(r->file_id, 1u);
            ASSERT_LE(r->file_id, 50u);
        }
    } TEST_END()

    TEST_BEGIN("arrival_times_monotonic") {
        pec::SyntheticEngine eng;
        int64_t prev = -1;
        for (int i = 0; i < 1000; ++i) {
            auto r = eng.generate();
            ASSERT_TRUE(r.has_value());
            ASSERT_GE(r->arrival_ns, prev);
            prev = r->arrival_ns;
        }
    } TEST_END()

    TEST_BEGIN("zipf_rank1_most_frequent") {
        pec::SyntheticConfig cfg;
        cfg.catalog_size = 100;
        cfg.zipf_alpha = 0.8;
        cfg.seed = 123;

        pec::SyntheticEngine eng(cfg);
        std::map<uint64_t, int> freq;
        for (int i = 0; i < 100000; ++i) {
            auto r = eng.generate();
            freq[r->file_id]++;
        }
        // File 1 (rank 1) should be the most frequent.
        int max_count = 0;
        uint64_t max_id = 0;
        for (auto& [id, cnt] : freq) {
            if (cnt > max_count) { max_count = cnt; max_id = id; }
        }
        ASSERT_EQ(max_id, 1u);
    } TEST_END()

    return test::test_main_return();
}
