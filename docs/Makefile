.PHONY: apidoc html clean all

apidoc:
	sphinx-apidoc -f -o . ../nebula

html:
	sphinx-build -b html . _build

clean:
	rm -rf _build

all: apidoc html
