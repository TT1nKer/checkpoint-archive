// The exponent of each predictor word is known to the decoder. In each exponent
// context, a single rANS symbol represents residual bit length and up to two
// leading payload bits. The leading one is implicit. Remaining bits are raw.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
using Bytes = std::vector<uint8_t>;
constexpr uint32_t scale_bits = 12;
constexpr uint32_t scale = 1 << scale_bits;
constexpr uint32_t lower = 1 << 23;
constexpr unsigned alphabet = 124;  // up to 32 bits: four symbols per length >= 3
struct Model {
    std::array<uint32_t, alphabet> counts{};
    std::array<uint16_t, alphabet> freq{}, start{};
};

void put16(Bytes& b, unsigned v) {
    b.push_back(v); b.push_back(v >> 8);
}
void put32(Bytes& b, uint32_t v) {
    put16(b, v); put16(b, v >> 16);
}

struct Reader {
    const uint8_t *p, *end;
    unsigned byte() {
        if (p == end) throw std::runtime_error("truncated stream");
        return *p++;
    }
    unsigned u16() {
        unsigned lo = byte();
        return lo | (byte() << 8);
    }
    uint32_t u32() {
        uint32_t lo = u16();
        return lo | (uint32_t(u16()) << 16);
    }
};

struct BitWriter {
    uint64_t buffer = 0;
    unsigned used = 0;
    Bytes bytes;
    void put(uint32_t value, unsigned width) {
        buffer |= uint64_t(value & ((uint32_t(1) << width) - 1)) << used;
        used += width;
        while (used >= 8) {
            bytes.push_back(buffer);
            buffer >>= 8;
            used -= 8;
        }
    }
    void finish() { if (used) bytes.push_back(buffer); }
};

struct BitReader {
    Reader source;
    uint64_t buffer = 0;
    unsigned used = 0;
    uint32_t get(unsigned width) {
        while (used < width) {
            buffer |= uint64_t(source.byte()) << used;
            used += 8;
        }
        const uint32_t out = buffer & ((uint32_t(1) << width) - 1);
        buffer >>= width;
        used -= width;
        return out;
    }
};

unsigned symbol(uint32_t z) {
    if (z < 4) return z;
    const unsigned width = 32u - __builtin_clz(z);
    return 4 * (width - 2) + ((z >> (width - 3)) & 3);
}

unsigned tail_width(unsigned s) { return s < 4 ? 0 : (s / 4 - 1); }

template<class Word> unsigned exponent(Word w) {
    if constexpr (sizeof(Word) == 2) return (w >> 10) & 31;
    return (w >> 23) & 255;
}

// Frequencies are positive for every observed symbol and sum exactly to scale.
void normalize(Model& m, uint64_t total, unsigned present) {
    std::vector<std::pair<uint64_t, unsigned>> remainders;
    unsigned assigned = 0;
    for (unsigned s = 0; s < alphabet; ++s) if (m.counts[s]) {
        const uint64_t numerator = uint64_t(m.counts[s]) * (scale - present);
        m.freq[s] = 1 + numerator / total;
        assigned += m.freq[s];
        remainders.emplace_back(numerator % total, s);
    }
    std::sort(remainders.rbegin(), remainders.rend());
    for (unsigned i = 0; i < scale - assigned; ++i) ++m.freq[remainders[i].second];
    unsigned start = 0;
    for (unsigned s = 0; s < alphabet; ++s) {
        m.start[s] = start;
        start += m.freq[s];
    }
}

template<class Word>
Bytes encode(const Word* zig, const Word* ref, size_t n) {
    constexpr unsigned contexts = sizeof(Word) == 2 ? 32 : 256;
    std::vector<Model> models(contexts);
    BitWriter tails;
    for (size_t i = 0; i < n; ++i) {
        const unsigned s = symbol(zig[i]);
        ++models[exponent(ref[i])].counts[s];
        tails.put(zig[i], tail_width(s));
    }
    tails.finish();
    Bytes out(2);
    unsigned used_contexts = 0;
    for (unsigned c = 0; c < contexts; ++c) {
        Model& m = models[c];
        unsigned present = 0;
        uint64_t total = 0;
        for (uint32_t cnt : m.counts) {
            total += cnt;
            present += cnt != 0;
        }
        if (!present) continue;
        ++used_contexts;
        normalize(m, total, present);
        out.push_back(c); out.push_back(present);
        for (unsigned s = 0; s < alphabet; ++s) if (m.freq[s]) {
            out.push_back(s); put16(out, m.freq[s]);
        }
    }
    out[0] = used_contexts; out[1] = used_contexts >> 8;
    uint32_t state = lower;
    Bytes reversed;
    for (size_t i = n; i-- > 0;) {
        const unsigned s = symbol(zig[i]);
        const auto& m = models[exponent(ref[i])];
        const uint32_t f = m.freq[s];
        const uint32_t threshold = ((lower >> scale_bits) * 256) * f;
        while (state >= threshold) {
            reversed.push_back(state);
            state >>= 8;
        }
        state = ((state / f) << scale_bits) + (state % f) + m.start[s];
    }
    if (reversed.size() > UINT32_MAX) throw std::length_error("rANS stream exceeds 4 GiB");
    put32(out, reversed.size());
    put32(out, state);
    out.insert(out.end(), reversed.rbegin(), reversed.rend());
    out.insert(out.end(), tails.bytes.begin(), tails.bytes.end());
    return out;
}

template<class Word>
void decode(const uint8_t* data, size_t len, const Word* ref, Word* zig, size_t n) {
    constexpr unsigned contexts = sizeof(Word) == 2 ? 32 : 256;
    std::vector<Model> models(contexts);
    std::vector<std::array<uint8_t, scale>> lookup(contexts);
    std::array<bool, contexts> seen{};
    Reader in{data, data + len};
    const unsigned count = in.u16();
    if (count > contexts) throw std::runtime_error("too many contexts");
    for (unsigned j = 0; j < count; ++j) {
        const unsigned c = in.byte(), present = in.byte();
        if (c >= contexts || seen[c] || !present || present > alphabet)
            throw std::runtime_error("invalid context");
        seen[c] = true;
        unsigned start = 0;
        auto& m = models[c];
        for (unsigned k = 0; k < present; ++k) {
            const unsigned s = in.byte(), f = in.u16();
            if (s >= (sizeof(Word) == 2 ? 60u : alphabet) || !f || m.freq[s] || start + f > scale)
                throw std::runtime_error("invalid frequency");
            m.freq[s] = f; m.start[s] = start;
            std::fill_n(lookup[c].begin() + start, f, s);
            start += f;
        }
        if (start != scale) throw std::runtime_error("unnormalized table");
    }
    const uint32_t rans_size = in.u32();
    uint32_t state = in.u32();
    if (state < lower || size_t(in.end - in.p) < rans_size) throw std::runtime_error("invalid state");
    Reader entropy{in.p, in.p + rans_size};
    BitReader tails{{entropy.end, in.end}};
    for (size_t i = 0; i < n; ++i) {
        const unsigned c = exponent(ref[i]);
        if (!seen[c]) throw std::runtime_error("missing context");
        const unsigned slot = state & (scale - 1);
        const unsigned s = lookup[c][slot];
        auto& m = models[c];
        state = m.freq[s] * (state >> scale_bits) + slot - m.start[s];
        while (state < lower) state = (state << 8) | entropy.byte();
        const unsigned bits = tail_width(s);
        zig[i] = s < 4 ? s : (uint32_t(4 + (s & 3)) << bits) | tails.get(bits);
    }
    if (state != lower || entropy.p != entropy.end || tails.source.p != tails.source.end || tails.buffer)
        throw std::runtime_error("incomplete stream");
}
}  // namespace

extern "C" void* pack_context(const void* z, const void* ref, size_t n, int width, size_t* len) {
    try {
        if ((width != 2 && width != 4) || n > UINT32_MAX) return nullptr;
        Bytes out = width == 2 ? encode((const uint16_t*)z, (const uint16_t*)ref, n)
                               : encode((const uint32_t*)z, (const uint32_t*)ref, n);
        void* result = std::malloc(out.size());
        if (!result) return nullptr;
        *len = out.size();
        std::memcpy(result, out.data(), out.size());
        return result;
    } catch (...) { return nullptr; }
}
extern "C" void release_context(void* p) { std::free(p); }
extern "C" int unpack_context(const uint8_t* data, size_t len, const void* ref, void* z,
                              size_t n, int width) {
    try {
        if (width == 2) decode(data, len, (const uint16_t*)ref, (uint16_t*)z, n);
        else if (width == 4) decode(data, len, (const uint32_t*)ref, (uint32_t*)z, n);
        else return 0;
        return 1;
    } catch (...) { return 0; }
}
