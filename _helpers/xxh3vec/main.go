// Command xxh3vec prints the XXH3-64 test vectors in crates/minlz/src/xxh3/vectors.txt,
// computed by github.com/zeebo/xxh3, the hash MinLZ Go uses for SIMD stream chunks.
//
//	go run . > ../../crates/minlz/src/xxh3/vectors.txt
//
// Input of length n is gen(n), the first n bytes of the xorshift64 stream below.
// Lines:
//
//	h <len> <hash>           xxh3.Hash
//	s <len> <hash> <sizes>   xxh3.Hasher fed Write calls of the comma-separated sizes, repeated, then Sum64
package main

import (
	"fmt"
	"strings"

	"github.com/zeebo/xxh3"
)

// gen returns n bytes: the little-endian bytes of successive xorshift64 (13, 7, 17) states.
func gen(n int) []byte {
	b := make([]byte, n)
	x := uint64(0x9E3779B97F4A7C15)
	for i := range b {
		if i%8 == 0 {
			x ^= x << 13
			x ^= x >> 7
			x ^= x << 17
		}
		b[i] = byte(x >> (8 * (i % 8)))
	}
	return b
}

func main() {
	src := gen(3 << 20)
	fmt.Println("# XXH3-64 vectors from _helpers/xxh3vec (github.com/zeebo/xxh3 v1.1.1-0.20260908150653-f7225998cebf).")
	lens := []int{}
	for n := 0; n <= 4096; n++ {
		lens = append(lens, n)
	}
	lens = append(lens, 4096+17, 8191, 8192, 8193, 16385, 65535, 65536, 65536+7, 100000, 1<<20, 1<<20+13, 3<<20)
	for _, n := range lens {
		fmt.Printf("h %d %016x\n", n, xxh3.Hash(src[:n]))
	}

	splits := [][]int{
		{1}, {7}, {63, 65}, {64}, {0, 1023, 1}, {1024}, {1025}, {1087, 1, 2}, {1088}, {1089},
		{960, 64}, {4096, 3}, {65536}, {100, 2000, 1, 5000, 64, 1024, 0, 1088, 70000},
	}
	slens := []int{0, 1, 64, 240, 241, 1024, 1025, 1088, 1089, 2048, 2112, 2113, 4096 + 17, 65536 + 7, 1<<20 + 13, 3 << 20}
	for _, n := range slens {
		for _, sp := range splits {
			minSize := n
			for _, s := range sp {
				if s > 0 {
					minSize = min(minSize, s)
				}
			}
			// Keeps the Rust debug-build tests quick.
			if minSize > 0 && n/minSize > 100000 {
				continue
			}
			h := xxh3.New()
			for i, k := 0, 0; i < n; k++ {
				s := min(sp[k%len(sp)], n-i)
				h.Write(src[i : i+s])
				i += s
			}
			sizes := strings.Trim(strings.ReplaceAll(fmt.Sprint(sp), " ", ","), "[]")
			fmt.Printf("s %d %016x %s\n", n, h.Sum64(), sizes)
		}
	}
}
