// Globally optimal directed single-parent checkpoint archive with bounded restore depth.
// Compile: c++ -O3 -std=c++17 forest.cpp -o checkpoint_forest
// Input: n K, n anchor prices, then n x n directed record prices [child][parent].
// A price of -1 forbids an edge. Diagonals are ignored. All other prices are integer bytes.
// Output: cost, then a space-separated parent array (-1 denotes an anchor).
// --all outputs those two lines for every depth 1..K in order.
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

using Bytes = std::int64_t;
using Mask = unsigned;

struct ArchiveDP {
    int n;
    Mask count;
    Mask all;
    Bytes infinity;
    std::vector<Bytes> anchor;
    std::vector<Bytes> edge;
    std::vector<Bytes> cheapest;
    std::vector<Bytes> previous;
    std::vector<std::vector<Mask>> chosen;

    ArchiveDP(int size, int max_depth)
        : n(size), count(Mask(1) << n), all(count - 1),
          infinity(std::numeric_limits<Bytes>::max() / (n + 2)),
          anchor(n), edge(n * n), cheapest(static_cast<std::size_t>(count) * n),
          previous(count), chosen(max_depth + 1) {}

    bool read_prices() {
        for (Bytes& price : anchor) {
            if (!(std::cin >> price) || price < 0 || price >= infinity) return false;
        }
        for (int child = 0; child < n; ++child) {
            for (int parent = 0; parent < n; ++parent) {
                Bytes& price = edge[child * n + parent];
                if (!(std::cin >> price) || price < -1 || price >= infinity) return false;
                if (child == parent || price == -1) price = infinity;
            }
        }
        return true;
    }

    void initialize() {
        // cheapest[T,i] is the cost of storing i in full or pointing i to a parent in T.
        std::copy(anchor.begin(), anchor.end(), cheapest.begin());
        for (Mask set = 1; set < count; ++set) {
            const int parent = __builtin_ctz(set);
            const Mask rest = set & (set - 1);
            previous[set] = previous[rest] + anchor[parent];
            for (int child = 0; child < n; ++child) {
                cheapest[set * n + child] =
                    std::min(cheapest[rest * n + child], edge[child * n + parent]);
            }
        }
    }

    void solve_layer(int depth, bool final_only) {
        // F_d(S): choose a shallow subset T, then attach each node of S minus T to T
        // or store it in full. Choosing T as all nodes above the last level covers
        // every feasible forest. Conversely, each construction respects the depth bound.
        std::vector<Bytes> current(previous);
        auto& shallow = chosen[depth];
        shallow.resize(count);
        std::iota(shallow.begin(), shallow.end(), Mask(0));
        std::vector<Bytes> subtotal(count);
        for (Mask set = 0; set < count; ++set) {
            const Bytes* attach = &cheapest[set * n];
            const Mask available = all ^ set;
            if (final_only) {
                Bytes cost = previous[set];
                for (Mask bits = available; bits; bits &= bits - 1) {
                    cost += attach[__builtin_ctz(bits)];
                }
                if (cost < current[all]) {
                    current[all] = cost;
                    shallow[all] = set;
                }
                continue;
            }
            // Ascending submask order makes the subtotal without the lowest bit available.
            for (Mask added = 0; added != available;) {
                added = (added - available) & available;
                subtotal[added] = subtotal[added & (added - 1)] + attach[__builtin_ctz(added)];
                const Mask combined = set | added;
                const Bytes cost = previous[set] + subtotal[added];
                if (cost < current[combined]) {
                    current[combined] = cost;
                    shallow[combined] = set;
                }
            }
        }
        previous.swap(current);
    }

    void print_solution(int depth) const {
        std::vector<int> parents(n, -1);
        Mask subset = all;
        for (int level = depth; level >= 2; --level) {
            const Mask shallow = chosen[level][subset];
            for (Mask bits = subset ^ shallow; bits; bits &= bits - 1) {
                const int child = __builtin_ctz(bits);
                Bytes best = anchor[child];
                for (Mask options = shallow; options; options &= options - 1) {
                    const int parent = __builtin_ctz(options);
                    const Bytes price = edge[child * n + parent];
                    if (price < best) {
                        best = price;
                        parents[child] = parent;
                    }
                }
            }
            subset = shallow;
        }
        std::cout << previous[all] << '\n';
        for (int i = 0; i < n; ++i) {
            std::cout << parents[i] << (i + 1 == n ? '\n' : ' ');
        }
    }
};

int main(int argc, char** argv) {
    const bool print_all = argc == 2 && std::string(argv[1]) == "--all";
    if (argc > 1 && !print_all) return 1;
    int n, k;
    if (!(std::cin >> n >> k) || n < 1 || n > 18 || k < 1 || k > n) return 1;
    ArchiveDP archive(n, k);
    if (!archive.read_prices()) return 1;
    archive.initialize();
    if (print_all || k == 1) archive.print_solution(1);
    for (int depth = 2; depth <= k; ++depth) {
        archive.solve_layer(depth, depth == k);
        if (print_all || depth == k) archive.print_solution(depth);
    }
}
