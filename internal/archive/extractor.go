package archive

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/bodgit/sevenzip"
	"github.com/mholt/archiver/v3"
)

// Extract unpacks an archive into destinationDir and returns all the absolute paths of the extracted files.
// If the file is not an archive (e.g. .pdf), it simply returns the input file path as a single-element slice.
func Extract(archivePath, destinationDir string) ([]string, error) {
	ext := strings.ToLower(filepath.Ext(archivePath))

	// Some compound extensions like .tar.gz need special checking
	if strings.HasSuffix(strings.ToLower(archivePath), ".tar.gz") {
		ext = ".tar.gz"
	}

	var isArchive bool
	switch ext {
	case ".zip", ".tar", ".gz", ".tgz", ".tar.gz", ".rar", ".bz2", ".xz", ".lz4", ".sz":
		isArchive = true
		if err := archiver.Unarchive(archivePath, destinationDir); err != nil {
			return nil, fmt.Errorf("archiver failed to extract %s: %w", ext, err)
		}
	case ".7z":
		isArchive = true
		if err := extract7z(archivePath, destinationDir); err != nil {
			return nil, fmt.Errorf("sevenzip failed to extract: %w", err)
		}
	default:
		// Not an archive (e.g. .pdf, .txt, .docx), just return the original file
		isArchive = false
	}

	if !isArchive {
		return []string{archivePath}, nil
	}

	// Walk destinationDir to find all extracted files
	var extractedFiles []string
	err := filepath.Walk(destinationDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			// Skip the original archive file itself
			if path == archivePath {
				return nil
			}

			// Basic ZipSlip protection verification (archiver handles this mostly, but good to be sure)
			if strings.Contains(path, "..") {
				return fmt.Errorf("malicious path detected in extraction: %s", path)
			}
			extractedFiles = append(extractedFiles, path)
		}
		return nil
	})

	return extractedFiles, err
}

func extract7z(archivePath, destinationDir string) error {
	r, err := sevenzip.OpenReader(archivePath)
	if err != nil {
		return err
	}
	defer r.Close()

	for _, f := range r.File {
		if strings.Contains(f.Name, "..") {
			continue // ZipSlip protection
		}

		extractedFilePath := filepath.Join(destinationDir, f.Name)
		if f.FileInfo().IsDir() {
			os.MkdirAll(extractedFilePath, f.Mode())
			continue
		}

		if err := os.MkdirAll(filepath.Dir(extractedFilePath), 0755); err != nil {
			return err
		}

		dstFile, err := os.OpenFile(extractedFilePath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
		if err != nil {
			return err
		}

		srcFile, err := f.Open()
		if err != nil {
			dstFile.Close()
			return err
		}

		_, err = io.Copy(dstFile, srcFile)
		dstFile.Close()
		srcFile.Close()
		if err != nil {
			return err
		}
	}
	return nil
}
