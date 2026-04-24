// bfd_pack.cpp — Thread-safe Best-Fit Decreasing bin packing.
//
// Compile: g++ -O3 -shared -fPIC -o libbfd_pack.so bfd_pack.cpp
//
// Algorithm: segment tree over remaining-capacity values [0..capacity],
// with a min-heap of bin IDs at each capacity level.  Finding the best
// bin for a document of size s costs O(log C) where C = capacity,
// regardless of how many bins are open.
//
// Thread-safe: all state is local to each bfd_pack() call.
//
// max_docs_per_bin: when >0, bins are withheld from the segment tree once
// they reach the cap, clamping the upper tail of docs-per-sample. Pass 0
// to disable.

#include <vector>
#include <queue>
#include <cstring>

// Tracks which remaining-capacity values have at least one open bin.
// Each leaf corresponds to one capacity value; internal nodes store the
// OR of their children (1 = at least one non-empty bucket below).

class SegmentTree {
    int n_;                                             // number of leaves (power of 2)
    std::vector<int> tree_;                             // 1-indexed, size 4*n_
    std::vector<std::priority_queue<int,              // min-heap per capacity value
                std::vector<int>, std::greater<int>>> buckets_;

public:
    explicit SegmentTree(int capacity) {
        // Round up to next power of 2
        n_ = 1;
        while (n_ <= capacity) n_ <<= 1;
        tree_.assign(4 * n_, 0);
        buckets_.resize(n_);
    }

    // Find the smallest remaining capacity >= target that has an open bin.
    // Returns -1 if none exists.
    int find_best_fit(int target) const {
        return query(target, 1, 0, n_ - 1);
    }

    // Add a bin to the bucket for the given remaining capacity.
    void add_bin(int remaining, int bin_id) {
        buckets_[remaining].push(bin_id);
        update(remaining, 1, 0, n_ - 1);
    }

    // Remove and return the smallest bin_id from the given capacity bucket.
    int pop_bin(int remaining) {
        int bin_id = buckets_[remaining].top();
        buckets_[remaining].pop();
        update(remaining, 1, 0, n_ - 1);
        return bin_id;
    }

private:
    void update(int pos, int node, int lo, int hi) {
        if (lo == hi) {
            tree_[node] = buckets_[pos].empty() ? 0 : 1;
            return;
        }
        int mid = (lo + hi) / 2;
        if (pos <= mid) update(pos, 2 * node,     lo, mid);
        else            update(pos, 2 * node + 1, mid + 1, hi);
        tree_[node] = tree_[2 * node] | tree_[2 * node + 1];
    }

    int query(int target, int node, int lo, int hi) const {
        if (!tree_[node] || hi < target) return -1;
        if (lo == hi) return lo;
        int mid = (lo + hi) / 2;
        int left = query(target, 2 * node, lo, mid);
        if (left != -1) return left;
        return query(target, 2 * node + 1, mid + 1, hi);
    }
};


// extern "C" so Python ctypes can call it by name.

extern "C" void bfd_pack(
    const int  *sorted_positions,   // indices into doc_lengths, sorted by decreasing length
    const long *doc_lengths,        // length of each document (indexed by position)
    int         num_docs,           // total number of documents
    int         capacity,           // bin capacity (seq_length + extra_token)
    int         max_docs_per_bin,   // <=0 means no cap
    const int  *document_index,     // maps position -> actual document ID
    // outputs:
    int        *doc_idx_out,        // reordered document IDs  (size: num_docs)
    int        *boundaries_out,     // bin boundary offsets     (size: num_docs + 1, worst case)
    int        *num_bins_out        // number of bins created
) {
    SegmentTree tree(capacity);
    std::vector<std::vector<int>> bins;     // bins[bin_id] = list of positions

    // A bin is eligible for more docs iff it has room AND is below the cap.
    auto try_insert_bin = [&](int bin_id, int new_remaining) {
        if (new_remaining <= 0) return;
        if (max_docs_per_bin > 0 &&
            static_cast<int>(bins[bin_id].size()) >= max_docs_per_bin) return;
        tree.add_bin(new_remaining, bin_id);
    };

    for (int i = 0; i < num_docs; i++) {
        int  pos    = sorted_positions[i];
        long length = doc_lengths[pos];

        // Oversized document: gets its own bin
        if (length > capacity) {
            bins.push_back({pos});
            continue;
        }

        // Zero-length document: tuck into any existing bin, or open a new one
        if (length == 0) {
            int r = tree.find_best_fit(0);
            if (r >= 0) {
                int bid = tree.pop_bin(r);
                bins[bid].push_back(pos);
                try_insert_bin(bid, r);             // remaining unchanged
            } else {
                int bid = static_cast<int>(bins.size());
                bins.push_back({pos});
                try_insert_bin(bid, capacity);
            }
            continue;
        }

        // Normal case: find tightest-fitting bin, or open a new one
        int r = tree.find_best_fit(static_cast<int>(length));
        if (r >= 0) {
            int bid   = tree.pop_bin(r);
            int new_r = r - static_cast<int>(length);
            bins[bid].push_back(pos);
            try_insert_bin(bid, new_r);
        } else {
            int bid   = static_cast<int>(bins.size());
            int new_r = capacity - static_cast<int>(length);
            bins.push_back({pos});
            try_insert_bin(bid, new_r);
        }
    }

    // Flatten bins into contiguous output arrays
    int offset = 0;
    for (int b = 0; b < static_cast<int>(bins.size()); b++) {
        boundaries_out[b] = offset;
        for (int pos : bins[b]) {
            doc_idx_out[offset++] = document_index[pos];
        }
    }
    boundaries_out[bins.size()] = offset;
    *num_bins_out = static_cast<int>(bins.size());
}