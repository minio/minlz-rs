// Command simdgo runs the Go MinLZ block codec for the Rust port's cross checks, and writes its SIMD fixtures.
//
//	simdgo enc -level N IN OUT    minlz.Encode at Go's int level N (4-7, 260-262 = SIMD|LevelSIMDDelta)
//	simdgo dec IN OUT             minlz.Decode
//	simdgo fixtures [-text FILE] DIR
//
// See _SIMD_PORT.md and crates/minlz/tests/simd_fixtures.rs.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/minio/minlz"
)

func usage() {
	fmt.Fprintln(os.Stderr, "usage: simdgo enc -level N IN OUT | dec IN OUT | fixtures [-text FILE] DIR")
	os.Exit(2)
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "simdgo:", err)
		os.Exit(1)
	}
}

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	fs := flag.NewFlagSet(os.Args[1], flag.ExitOnError)
	level := fs.Int("level", minlz.LevelBalancedSIMD, "Go level")
	text := fs.String("text", "testdata/Mark.Twain-Tom.Sawyer.txt", "Tom Sawyer text, for fixtures")
	check(fs.Parse(os.Args[2:]))
	switch {
	case os.Args[1] == "enc" && fs.NArg() == 2:
		in, err := os.ReadFile(fs.Arg(0))
		check(err)
		out, err := minlz.Encode(nil, in, *level)
		check(err)
		check(os.WriteFile(fs.Arg(1), out, 0o644))
	case os.Args[1] == "dec" && fs.NArg() == 2:
		in, err := os.ReadFile(fs.Arg(0))
		check(err)
		out, err := minlz.Decode(nil, in)
		check(err)
		check(os.WriteFile(fs.Arg(1), out, 0o644))
	case os.Args[1] == "fixtures" && fs.NArg() == 1:
		t, err := os.ReadFile(*text)
		check(err)
		check(fixtures(fs.Arg(0), t))
	default:
		usage()
	}
}
