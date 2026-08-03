package processor

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// Helper function to dynamically construct a valid minimal PDF with accurate xref offsets
func buildValidTestPDF(streamContent string) []byte {
	var buf bytes.Buffer
	buf.WriteString("%PDF-1.4\n")

	offsets := make([]int, 7)

	offsets[1] = buf.Len()
	buf.WriteString("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")

	offsets[2] = buf.Len()
	buf.WriteString("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")

	offsets[3] = buf.Len()
	buf.WriteString("3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> /XObject << /Im1 6 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>\nendobj\n")

	offsets[4] = buf.Len()
	buf.WriteString("4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

	offsets[5] = buf.Len()
	fmt.Fprintf(&buf, "5 0 obj\n<< /Length %d >>\nstream\n%s\nendstream\nendobj\n", len(streamContent), streamContent)

	offsets[6] = buf.Len()
	buf.WriteString("6 0 obj\n<< /Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Length 3 >>\nstream\n\xff\xff\xff\nendstream\nendobj\n")

	startXRef := buf.Len()
	buf.WriteString("xref\n0 7\n")
	buf.WriteString("0000000000 65535 f \n")
	for i := 1; i <= 6; i++ {
		fmt.Fprintf(&buf, "%010d 00000 n \n", offsets[i])
	}
	fmt.Fprintf(&buf, "trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF", startXRef)

	return buf.Bytes()
}

func TestChallenger_DetectIsScannedPDF_AdversarialSuite(t *testing.T) {
	tempDir := t.TempDir()

	tests := []struct {
		name          string
		streamContent string
		customPath    string
		wantIsScanned bool
		wantErr       bool
	}{
		{
			name: "StructuredText_LiteralTj",
			streamContent: `BT
/F1 12 Tf
100 700 Td
(Hello World) Tj
ET`,
			wantIsScanned: false,
			wantErr:       false,
		},
		{
			name: "StructuredText_HexTj",
			streamContent: `BT
/F1 12 Tf
100 700 Td
<48656c6c6f20576f726c64> Tj
ET`,
			wantIsScanned: false,
			wantErr:       false,
		},
		{
			name: "StructuredText_ArrayTJ_Literal",
			streamContent: `BT
/F1 12 Tf
100 700 Td
[(Hello) -10 (World)] TJ
ET`,
			wantIsScanned: false,
			wantErr:       false,
		},
		{
			name: "StructuredText_ArrayTJ_Hex",
			streamContent: `BT
/F1 12 Tf
100 700 Td
[<48656c6c6f>] TJ
ET`,
			wantIsScanned: false,
			wantErr:       false,
		},
		{
			name: "StructuredText_SingleQuoteOperator",
			streamContent: `BT
/F1 12 Tf
(Line of text) '
ET`,
			wantIsScanned: false,
			wantErr:       false,
		},
		{
			name: "StructuredText_DoubleQuoteOperator",
			streamContent: `BT
/F1 12 Tf
2 5 (Line of text) "
ET`,
			wantIsScanned: false,
			wantErr:       false,
		},
		{
			name: "ScannedImage_OnlyImageXObject",
			streamContent: `q
1 0 0 1 0 0 cm
/Im1 Do
Q`,
			wantIsScanned: true,
			wantErr:       false,
		},
		{
			name: "WhitespaceOnly_SpacesLiteralTj",
			streamContent: "BT\n/F1 12 Tf\n(       ) Tj\nET",
			wantIsScanned: true,
			wantErr:       false,
		},
		{
			name: "WhitespaceOnly_EscapedTabNewlineLiteralTj",
			streamContent: "BT\n/F1 12 Tf\n(   \\t\\n   ) Tj\nET",
			wantIsScanned: true,
			wantErr:       false,
		},
		{
			name: "WhitespaceOnly_HexTj",
			streamContent: `BT
/F1 12 Tf
<20202020> Tj
ET`,
			wantIsScanned: true,
			wantErr:       false,
		},
		{
			name: "WhitespaceOnly_ArrayTJ",
			streamContent: "BT\n/F1 12 Tf\n[(  ) 10 (   )] TJ\nET",
			wantIsScanned: true,
			wantErr:       false,
		},
		{
			name: "EmptyBTBlock",
			streamContent: `BT
ET`,
			wantIsScanned: true,
			wantErr:       false,
		},
		{
			name: "HybridPDF_ImageAndText",
			streamContent: `q
/Im1 Do
Q
BT
/F1 12 Tf
(Embedded text in hybrid document) Tj
ET`,
			wantIsScanned: false,
			wantErr:       false,
		},
		{
			name:          "CorruptedFile_InvalidPDFHeader",
			customPath:    filepath.Join(tempDir, "corrupt.pdf"),
			wantIsScanned: false,
			wantErr:       true,
		},
		{
			name:          "NonExistentFile",
			customPath:    filepath.Join(tempDir, "does_not_exist_123.pdf"),
			wantIsScanned: false,
			wantErr:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			targetPath := tt.customPath
			if targetPath == "" {
				pdfBytes := buildValidTestPDF(tt.streamContent)
				targetPath = filepath.Join(tempDir, tt.name+".pdf")
				if err := os.WriteFile(targetPath, pdfBytes, 0644); err != nil {
					t.Fatalf("Failed to create test PDF %s: %v", tt.name, err)
				}
			} else if tt.name == "CorruptedFile_InvalidPDFHeader" {
				if err := os.WriteFile(targetPath, []byte("NOT_A_PDF_HEADER_DATA"), 0644); err != nil {
					t.Fatalf("Failed to create corrupt PDF: %v", err)
				}
			}

			gotIsScanned, err := detectIsScannedPDF(targetPath)
			if (err != nil) != tt.wantErr {
				t.Fatalf("detectIsScannedPDF(%s) error = %v, wantErr %v", tt.name, err, tt.wantErr)
			}
			if !tt.wantErr && gotIsScanned != tt.wantIsScanned {
				t.Errorf("detectIsScannedPDF(%s) gotIsScanned = %v, wantIsScanned = %v", tt.name, gotIsScanned, tt.wantIsScanned)
			}
		})
	}
}
