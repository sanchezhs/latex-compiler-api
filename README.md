# Instructions

## Build the container
1) Build
docker build -t latex-compiler-api .

2) Run
docker run --rm -p 8000:8000 latex-compiler-api

## Pack the project

### ZIP your project first (include main.tex, images, .bib, etc.)
(You can download your project from Overleaf as ZIP also)

zip -r project.zip .

### Engine defaults to xelatex; return pdf by default
curl -X POST "http://localhost:8000/compile" \
  -F "project=@project.zip" \
  -F "main=main.tex" \
  -F "engine=xelatex" \
  -F "timeout_sec=300" \
  -o output.pdf -D -

