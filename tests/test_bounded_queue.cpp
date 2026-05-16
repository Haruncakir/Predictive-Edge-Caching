// test_bounded_queue.cpp ───────────────────────────────────────────────────

#include "test_harness.hpp"
#include "IOOptimizer.hpp"

#include <atomic>
#include <thread>
#include <vector>

int main() {
    std::cout << "=== BoundedQueue Tests ===\n";

    TEST_BEGIN("push_pop_single") {
        pec::BoundedQueue<int> q(4);
        q.push(42);
        int out = 0;
        ASSERT_TRUE(q.pop(out));
        ASSERT_EQ(out, 42);
    } TEST_END()

    TEST_BEGIN("push_pop_batch") {
        pec::BoundedQueue<int> q(8);
        for (int i = 0; i < 5; ++i) q.push(i);

        std::vector<int> batch;
        auto n = q.pop_batch(batch, 10);
        ASSERT_EQ(n, 5u);
        ASSERT_EQ(batch.size(), 5u);
        ASSERT_EQ(batch[0], 0);
        ASSERT_EQ(batch[4], 4);
    } TEST_END()

    TEST_BEGIN("close_unblocks_pop") {
        pec::BoundedQueue<int> q(4);
        std::thread closer([&]{
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            q.close();
        });
        int out = 0;
        ASSERT_FALSE(q.pop(out));  // should return false after close
        closer.join();
    } TEST_END()

    TEST_BEGIN("capacity_1_edge") {
        pec::BoundedQueue<int> q(1);
        q.push(99);
        ASSERT_EQ(q.size(), 1u);

        int out = 0;
        ASSERT_TRUE(q.pop(out));
        ASSERT_EQ(out, 99);
        ASSERT_EQ(q.size(), 0u);
    } TEST_END()

    TEST_BEGIN("producer_consumer_stress") {
        pec::BoundedQueue<int> q(16);
        const int N = 10000;
        std::atomic<int> sum{0};

        std::thread producer([&]{
            for (int i = 1; i <= N; ++i) q.push(i);
            q.close();
        });

        std::thread consumer([&]{
            int val = 0;
            while (q.pop(val)) sum.fetch_add(val);
        });

        producer.join();
        consumer.join();
        ASSERT_EQ(sum.load(), N * (N + 1) / 2);
    } TEST_END()

    return test::test_main_return();
}
