package server

import "os"

func init() {
	os.Setenv("MAX_PAGES_BEFORE_SPLIT", "10")
}
