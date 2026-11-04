package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/minio/minlz"
)

// fixtureLevels are Go's int levels: the SIMD levels, then the delta ones (LevelSIMDDelta = 256).
var fixtureLevels = []int{4, 5, 6, 7, 260, 261, 262}

// fixtureNames are the inputs, "kind-size". crates/minlz/tests/simd_fixtures.rs regenerates them from the name.
func fixtureNames() (names []string) {
	add := func(kind string, sizes ...int) {
		for _, s := range sizes {
			names = append(names, fmt.Sprintf("%s-%d", kind, s))
		}
	}
	add("textlike", 0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65, 66, 80, 96, 100, 112, 120,
		127, 128, 129, 140, 150, 160, 199, 200)
	add("zeros", 64, 72, 76, 80, 84, 88, 96, 128, 200, 65536)
	add("random", 65536)
	add("period3", 65536)
	add("period7", 65536)
	add("period31", 65536)
	add("period33", 65536)
	add("textlike", 2048, 2049, 4095, 4096, 4097, 65535, 65536, 65537)
	add("sparse", 512<<10)
	add("sawyer", 1024, 2048, 2049, 4095, 4096, 4097, 16384, 65536, 409600)
	add("mixed", 1100<<10)
	return names
}

const manifestHeader = `# SIMD block fixtures of the Go encoder (github.com/minio/minlz), written by _helpers/simdgo:
#   cd _helpers/simdgo && go run . fixtures -text ../../testdata/Mark.Twain-Tom.Sawyer.txt ../../crates/minlz/tests/fixtures/simd
# Remove the old .mzb files first. Inputs are regenerated from their names, see crates/minlz/tests/simd_fixtures.rs.
# Blocks of kind simd are in <input>.L<level>.mzb; stored blocks (0, 0, input; 0 if empty) have no file.
# input level insize encsize kind fnv1a64(input)
`

// fixtures writes the manifest and the SIMD blocks of every input and level to dir, checking that each decodes.
func fixtures(dir string, text []byte) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	m := []byte(manifestHeader)
	for _, name := range fixtureNames() {
		src, err := input(name, text)
		if err != nil {
			return err
		}
		h := fnv.New64a()
		h.Write(src)
		for _, level := range fixtureLevels {
			enc, err := minlz.Encode(nil, src, level)
			if err != nil {
				return err
			}
			if dec, err := minlz.Decode(nil, enc); err != nil || !bytes.Equal(dec, src) {
				return fmt.Errorf("%s level %d: does not decode: %v", name, level, err)
			}
			kind := "stored"
			if !bytes.Equal(enc, stored(src)) {
				kind = "simd"
				if v, n := binary.Uvarint(enc[1:]); enc[0] != 0 || n <= 0 || v>>24 != 1 {
					return fmt.Errorf("%s level %d: not a SIMD block", name, level)
				}
				if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("%s.L%d.mzb", name, level)), enc, 0o644); err != nil {
					return err
				}
			}
			m = fmt.Appendf(m, "%s %d %d %d %s %016x\n", name, level, len(src), len(enc), kind, h.Sum64())
		}
	}
	return os.WriteFile(filepath.Join(dir, "manifest.txt"), m, 0o644)
}

// stored is minlz's uncompressed block of src.
func stored(src []byte) []byte {
	if len(src) == 0 {
		return []byte{0}
	}
	return append([]byte{0, 0}, src...)
}

// input generates the input name, "kind-size", from an rng seeded with the FNV-1a hash of name.
func input(name string, text []byte) ([]byte, error) {
	i := strings.LastIndexByte(name, '-')
	size, err := strconv.Atoi(name[i+1:])
	if err != nil {
		return nil, fmt.Errorf("input %q: %w", name, err)
	}
	h := fnv.New64a()
	h.Write([]byte(name))
	r := rng(h.Sum64())
	kind := name[:i]
	switch {
	case kind == "zeros":
		return make([]byte, size), nil
	case kind == "random":
		return r.bytes(size), nil
	case kind == "textlike":
		return r.textLike(size), nil
	case kind == "sparse":
		return r.sparse(size), nil
	case strings.HasPrefix(kind, "period"):
		p, err := strconv.Atoi(kind[len("period"):])
		if err != nil {
			return nil, fmt.Errorf("input %q: %w", name, err)
		}
		return r.periodRuns(p, size), nil
	case kind == "sawyer":
		return r.sawyer(text, size), nil
	case kind == "mixed":
		return r.mixed(text, size), nil
	}
	return nil, fmt.Errorf("input %q: unknown kind", name)
}

// rng is SplitMix64.
type rng uint64

func (r *rng) next() uint64 {
	*r += 0x9e3779b97f4a7c15
	z := uint64(*r)
	z = (z ^ z>>30) * 0xbf58476d1ce4e5b9
	z = (z ^ z>>27) * 0x94d049bb133111eb
	return z ^ z>>31
}

func (r *rng) intn(n int) int { return int(r.next() % uint64(n)) }

func (r *rng) bytes(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(r.next())
	}
	return b
}

var words = strings.Fields("the and of to a in is it you that he was for on are with")

// word appends a random word of a small vocabulary and a separator to b.
func (r *rng) word(b []byte) []byte {
	b = append(b, words[r.intn(len(words))]...)
	if r.intn(8) == 0 {
		return append(b, '\n')
	}
	return append(b, ' ')
}

// words returns random words.
func (r *rng) words(size int) []byte {
	var b []byte
	for len(b) < size {
		b = r.word(b)
	}
	return b[:size]
}

// textLike returns phrases of 1 to 8 random words, each repeated 1 to 8 times.
func (r *rng) textLike(size int) []byte {
	var b []byte
	for len(b) < size {
		var p []byte
		for range 1 + r.intn(8) {
			p = r.word(p)
		}
		for range 1 + r.intn(8) {
			b = append(b, p...)
		}
	}
	return b[:size]
}

// sparse returns a random pattern of 7 bytes, each time followed by one of 2 random bytes.
// It codes many operations cheaply, so the block has several chunks, which can reuse tables.
func (r *rng) sparse(size int) []byte {
	pat, alpha := r.bytes(7), r.bytes(2)
	var b []byte
	for len(b) < size {
		b = append(append(b, pat...), alpha[r.intn(2)])
	}
	return b[:size]
}

// periodRuns returns runs of 64 to 4159 bytes, each repeating a random pattern of period bytes.
func (r *rng) periodRuns(period, size int) []byte {
	var b []byte
	for len(b) < size {
		pat := r.bytes(period)
		for i := range 64 + r.intn(4096) {
			b = append(b, pat[i%period])
		}
	}
	return b[:size]
}

// sawyer returns text, then runs of 1 to 8 lines of it, a quarter of the runs with a random lowercase letter.
func (r *rng) sawyer(text []byte, size int) []byte {
	var lines [][]byte
	for l := range bytes.Lines(text) {
		lines = append(lines, l)
	}
	b := bytes.Clone(text)
	for len(b) < size {
		start, first := len(b), r.intn(len(lines))
		for _, l := range lines[first:min(first+1+r.intn(8), len(lines))] {
			b = append(b, l...)
		}
		if r.intn(4) == 0 {
			i := start + r.intn(len(b)-start)
			b[i] = 'a' + byte(r.intn(26))
		}
	}
	return b[:size]
}

// mixed concatenates the kinds of data a large block meets: text, literals without matches, short periods,
// zeros, random data and many short matches of words. Then copies of earlier data at offsets of every octave,
// each followed by a random byte.
func (r *rng) mixed(text []byte, size int) []byte {
	b := r.sawyer(text, 128<<10)
	for range 48 << 10 {
		b = append(b, text[r.intn(len(text))])
	}
	for _, p := range []int{1, 2, 3, 4, 7, 8, 16, 31, 32, 33, 64} {
		b = append(b, r.periodRuns(p, 8<<10)...)
	}
	b = append(b, make([]byte, 128<<10)...)
	b = append(b, r.bytes(4<<10)...)
	b = append(b, r.words(96<<10)...)
	for len(b) < size {
		n := 256 + r.intn(8192)
		o := r.intn(len(b))
		off := 1 + o>>r.intn(20)
		for range n {
			b = append(b, b[len(b)-off])
		}
		b = append(b, byte(r.next()))
	}
	return b[:size]
}
