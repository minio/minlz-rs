module simdgo

go 1.25

require github.com/minio/minlz v0.0.0

require (
	github.com/klauspost/compress v1.20.1 // indirect
	github.com/klauspost/cpuid/v2 v2.4.0 // indirect
	github.com/zeebo/xxh3 v1.1.1-0.20260908150653-f7225998cebf // indirect
	golang.org/x/sys v0.41.0 // indirect
)

replace github.com/minio/minlz => ../../../minlz
