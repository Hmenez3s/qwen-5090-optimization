GO ?= go
VERSION ?= 0.1.0
DIST := dist
EXE := $(DIST)/qwenctl.exe
README := README.md

.PHONY: fmt test vet build package

fmt:
	$(GO)fmt -w main.go config.go manager.go

test:
	$(GO) test ./...
	PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_wsl_manager.py

vet:
	$(GO) vet ./...

build: fmt
	mkdir -p $(DIST)
	GOOS=windows GOARCH=amd64 CGO_ENABLED=0 $(GO) build -trimpath -ldflags="-s -w -X main.version=$(VERSION)" -o $(EXE) .
	cp $(README) $(DIST)/README.md
	cp Install-QwenCtl.ps1 $(DIST)/Install-QwenCtl.ps1

package: build
	cd $(DIST) && zip -q -9 qwenctl-windows-amd64.zip qwenctl.exe README.md Install-QwenCtl.ps1
	cd $(DIST) && shasum -a 256 qwenctl.exe qwenctl-windows-amd64.zip > SHA256SUMS
