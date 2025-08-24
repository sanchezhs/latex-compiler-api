import io
import os
import shutil
import tarfile
import tempfile
import zipfile
import subprocess
import re
import logging
from pathlib import Path
from typing import Optional, Literal, List, Dict, Any
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LaTeX Compiler API", version="1.2.0")

# Add CORS middleware for better web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Error Classification System --------


class ErrorSeverity(str, Enum):
    CRITICAL = "critical"  # Cannot continue, no PDF
    ERROR = "error"  # Serious issues but PDF might exist
    WARNING = "warning"  # Minor issues, PDF should be fine
    INFO = "info"  # Informational messages


class LaTeXError(BaseModel):
    severity: ErrorSeverity
    type: str
    message: str
    line_number: Optional[int] = None
    file: Optional[str] = None
    suggestion: Optional[str] = None


class CompilationResult(BaseModel):
    success: bool
    return_code: int
    pdf_generated: bool
    pdf_size: Optional[int] = None
    errors: List[LaTeXError] = []
    warnings: List[LaTeXError] = []
    execution_time: float
    pages_count: Optional[int] = None


def _classify_latex_errors(
    log_content: str, output_content: str = ""
) -> List[LaTeXError]:
    """Analyze LaTeX logs and classify errors with suggestions."""
    errors = []
    combined_content = f"{output_content}\n{log_content}"

    # Error patterns with classifications and suggestions
    error_patterns = [
        # Critical errors
        {
            "pattern": r"! LaTeX Error: File `([^']+)' not found",
            "severity": ErrorSeverity.CRITICAL,
            "type": "missing_file",
            "extract_info": lambda m: {
                "file": m.group(1),
                "suggestion": f"Add the missing file '{m.group(1)}' to your project or remove the reference",
            },
        },
        {
            "pattern": r"! Undefined control sequence\.\s*l\.(\d+)\s*(.+)",
            "severity": ErrorSeverity.ERROR,
            "type": "undefined_command",
            "extract_info": lambda m: {
                "line_number": int(m.group(1)),
                "suggestion": f"Check if you need to load a package for the command near line {m.group(1)}",
            },
        },
        {
            "pattern": r"! Package (\w+) Error: (.+)",
            "severity": ErrorSeverity.ERROR,
            "type": "package_error",
            "extract_info": lambda m: {
                "suggestion": f"Check the {m.group(1)} package documentation for: {m.group(2)}"
            },
        },
        {
            "pattern": r"! Unable to load picture or PDF file '([^']+)'",
            "severity": ErrorSeverity.ERROR,
            "type": "missing_image",
            "extract_info": lambda m: {
                "file": m.group(1),
                "suggestion": f"Add the missing image '{m.group(1)}' or check the file path",
            },
        },
        {
            "pattern": r"! Emergency stop",
            "severity": ErrorSeverity.CRITICAL,
            "type": "emergency_stop",
            "extract_info": lambda _: {
                "suggestion": "LaTeX encountered a fatal error. Check the lines above for the specific cause"
            },
        },
        # Warnings
        {
            "pattern": r"LaTeX Warning: (.+) on input line (\d+)",
            "severity": ErrorSeverity.WARNING,
            "type": "latex_warning",
            "extract_info": lambda m: {
                "line_number": int(m.group(2)),
                "suggestion": f"Consider addressing: {m.group(1)}",
            },
        },
        {
            "pattern": r"Package (\w+) Warning: (.+)",
            "severity": ErrorSeverity.WARNING,
            "type": "package_warning",
            "extract_info": lambda m: {
                "suggestion": f"Package {m.group(1)} warning: {m.group(2)}"
            },
        },
        {
            "pattern": r"Overfull \\hbox \([^)]+\) in paragraph at lines (\d+)--(\d+)",
            "severity": ErrorSeverity.WARNING,
            "type": "overfull_hbox",
            "extract_info": lambda m: {
                "line_number": int(m.group(1)),
                "suggestion": f"Text overflow between lines {m.group(1)}-{m.group(2)}. Consider rewording or using \\linebreak",
            },
        },
        {
            "pattern": r"Underfull \\vbox \(badness (\d+)\)",
            "severity": ErrorSeverity.WARNING,
            "type": "underfull_vbox",
            "extract_info": lambda m: {
                "suggestion": f"Page layout issue (badness {m.group(1)}). Usually harmless but consider adjusting content"
            },
        },
        # Bibliography warnings
        {
            "pattern": r"File '([^']+)' is wrong format version - expected ([0-9.]+)",
            "severity": ErrorSeverity.WARNING,
            "type": "bibliography_version",
            "extract_info": lambda m: {
                "file": m.group(1),
                "suggestion": f"Bibliography file format mismatch. Run biber/bibtex to regenerate {m.group(1)}",
            },
        },
        # Font warnings
        {
            "pattern": r"Font shape `([^']+)' undefined",
            "severity": ErrorSeverity.WARNING,
            "type": "font_warning",
            "extract_info": lambda m: {
                "suggestion": f"Font shape {m.group(1)} not found. LaTeX will substitute a similar font"
            },
        },
    ]

    for pattern_info in error_patterns:
        pattern = pattern_info["pattern"]
        for match in re.finditer(
            pattern, combined_content, re.MULTILINE | re.IGNORECASE
        ):
            error_data = {
                "severity": pattern_info["severity"],
                "type": pattern_info["type"],
                "message": match.group(0).strip(),
            }

            if "extract_info" in pattern_info:
                additional_info = pattern_info["extract_info"](match)
                error_data.update(additional_info)

            errors.append(LaTeXError(**error_data))

    return errors


def _extract_compilation_stats(log_content: str) -> Dict[str, Any]:
    """Extract useful statistics from LaTeX log."""
    stats = {}

    # Extract page count
    page_match = re.search(
        r"Output written on .+\.(?:pdf|xdv) \((\d+) pages?", log_content
    )
    if page_match:
        stats["pages"] = int(page_match.group(1))

    # Extract memory usage
    memory_match = re.search(r"(\d+) strings out of \d+", log_content)
    if memory_match:
        stats["strings_used"] = int(memory_match.group(1))

    # Extract font info
    font_match = re.search(r"(\d+) words of font info for (\d+) fonts", log_content)
    if font_match:
        stats["fonts_loaded"] = int(font_match.group(2))

    return stats


# -------- Utilities --------


def _extract_archive(archive_path: Path, dest_dir: Path):
    """Extract .zip or .tar(.gz/.bz2/.xz) into dest_dir."""
    suffixes = "".join(archive_path.suffixes)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif suffixes.endswith(
        (".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".tar")
    ):
        mode = "r"
        if suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz"):
            mode = "r:gz"
        elif suffixes.endswith(".tar.bz2") or suffixes.endswith(".tbz2"):
            mode = "r:bz2"
        elif suffixes.endswith(".tar.xz") or suffixes.endswith(".txz"):
            mode = "r:xz"
        with tarfile.open(archive_path, mode) as tf:
            tf.extractall(dest_dir)
    else:
        raise ValueError(
            "Unsupported archive format. Please upload a .zip or .tar.gz/.bz2/.xz"
        )


def _find_project_root(extracted_dir: Path) -> Path:
    """Find the actual project directory, handling cases where ZIP contains a single folder."""
    # List all items in the extracted directory
    items = list(extracted_dir.iterdir())

    # If there's only one item and it's a directory, that's probably the project root
    if len(items) == 1 and items[0].is_dir():
        return items[0]

    # Otherwise, the extracted directory is the project root
    return extracted_dir


def _guess_main_tex(project_dir: Path) -> Optional[Path]:
    """Try to find an entrypoint .tex if not provided."""
    print(f"Searching for main .tex file in: {project_dir}")

    # Priority: main.tex, thesis.tex, report.tex, else first *.tex found (alphabetically)
    candidates = ["main.tex", "thesis.tex", "tesis.tex", "report.tex", "paper.tex"]

    # First, look in the project root
    for c in candidates:
        p = project_dir / c
        if p.exists():
            print(f"Found candidate: {p}")
            return p

    # Then search recursively
    tex_files = sorted(project_dir.rglob("*.tex"))
    # Exclude common template fragment names
    tex_files = [t for t in tex_files if t.name not in {"preamble.tex"}]

    if tex_files:
        print(f"Found .tex files: {[str(f) for f in tex_files[:5]]}")  # Show first 5
        return tex_files[0]

    return None


def _build_cmd(
    engine: Literal["xelatex", "lualatex", "pdflatex"],
    force_biber: bool,
    main_file: Path,
    working_dir: Path,
) -> list[str]:
    """Construct latexmk command."""
    cmd = ["latexmk", "-interaction=nonstopmode", "-halt-on-error"]

    if engine == "xelatex":
        cmd += ["-xelatex"]
    elif engine == "lualatex":
        cmd += ["-lualatex"]
    else:
        cmd += ["-pdf"]  # pdflatex

    if force_biber:
        cmd += ["-biber"]

    # Use relative path from working directory
    rel_path = main_file.relative_to(working_dir)
    cmd += [str(rel_path)]

    return cmd


def _run(cmd: list[str], cwd: Path, timeout_sec: int) -> tuple[int, str]:
    """Run a subprocess and capture combined stdout/stderr."""
    logger.info(f"Running command: {' '.join(cmd)}")
    logger.info(f"Working directory: {cwd}")

    # Set up environment
    env = os.environ.copy()
    env["max_print_line"] = "1000"
    env["TEXMFHOME"] = str(cwd)  # Make LaTeX search in current directory for .sty files

    start_time = datetime.now()
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    try:
        out, _ = proc.communicate(timeout=timeout_sec)
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Command completed in {execution_time:.2f} seconds with return code {proc.returncode}"
        )
        return proc.returncode, out
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Command timed out after {execution_time:.2f} seconds")
        return 124, out + "\n[ERROR] Timeout exceeded"


def _find_output_pdf(
    project_dir: Path, main_file: Path, jobname: Optional[str] = None
) -> Optional[Path]:
    """Find the generated PDF file."""
    if jobname:
        # Look for jobname.pdf in project directory
        pdf_candidates = [
            project_dir / f"{jobname}.pdf",
            main_file.parent / f"{jobname}.pdf",
        ]
    else:
        # Look for main_file_stem.pdf
        stem = main_file.stem
        pdf_candidates = [main_file.parent / f"{stem}.pdf", project_dir / f"{stem}.pdf"]

    for pdf_path in pdf_candidates:
        if pdf_path.exists():
            print(f"Found PDF: {pdf_path}")
            return pdf_path

    # Fallback: search for any PDF files
    pdf_files = list(project_dir.rglob("*.pdf"))
    # Filter out obvious non-output PDFs
    pdf_files = [
        p
        for p in pdf_files
        if not any(
            x in p.name.lower() for x in ["cover", "titulo", "title", "backcover"]
        )
    ]

    if pdf_files:
        print(f"Found PDF files: {[str(f) for f in pdf_files]}")
        # Return the most recently modified one
        return max(pdf_files, key=lambda p: p.stat().st_mtime)

    return None


def _zip_with_logs(project_dir: Path, pdf_path: Path, build_log: str) -> bytes:
    """Create a zip containing the output PDF, relevant logs, and latexmk output."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if pdf_path.exists():
            zf.write(pdf_path, arcname=pdf_path.name)

        # Include all .log files
        for log_file in project_dir.rglob("*.log"):
            try:
                rel_path = log_file.relative_to(project_dir)
                zf.write(log_file, arcname=str(rel_path))
            except Exception:
                pass

        # Include other useful files
        for pattern in ["*.aux", "*.bbl", "*.blg", "*.fls", "*.fdb_latexmk"]:
            for file in project_dir.rglob(pattern):
                try:
                    rel_path = file.relative_to(project_dir)
                    zf.write(file, arcname=str(rel_path))
                except Exception:
                    pass

        # Include latexmk combined output
        zf.writestr("build_output.txt", build_log)

    buf.seek(0)
    return buf.read()


# -------- Models --------


class Health(BaseModel):
    ok: bool = True
    version: str = "1.2.0"
    timestamp: str


class CompilationStatus(BaseModel):
    project_structure: list[str]
    main_file_found: Optional[str]
    working_directory: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: CompilationResult
    timestamp: str
    suggestions: List[str] = []


# -------- Routes --------


@app.get("/health", response_model=Health)
def health():
    return Health(ok=True, version="1.2.0", timestamp=datetime.now().isoformat())


@app.post("/debug")
async def debug_project(
    project: UploadFile = File(..., description="ZIP or TAR of LaTeX project"),
    main: Optional[str] = Form(None, description="Main TeX file name"),
):
    """Debug endpoint to inspect project structure."""
    workdir = Path(tempfile.mkdtemp(prefix="latex-debug-"))
    try:
        # Save upload
        archive_path = workdir / f"upload{Path(project.filename).suffix}"
        with open(archive_path, "wb") as f:
            content = await project.read()
            f.write(content)

        # Extract
        extract_dir = workdir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        _extract_archive(archive_path, extract_dir)

        # Find project root
        project_dir = _find_project_root(extract_dir)

        # List all files
        all_files = []
        for item in project_dir.rglob("*"):
            if item.is_file():
                try:
                    rel_path = item.relative_to(project_dir)
                    all_files.append(str(rel_path))
                except Exception:
                    all_files.append(str(item))

        # Find main file
        main_file = None
        if main:
            potential_main = project_dir / main
            if potential_main.exists():
                main_file = str(potential_main.relative_to(project_dir))
        else:
            found_main = _guess_main_tex(project_dir)
            if found_main:
                main_file = str(found_main.relative_to(project_dir))

        return CompilationStatus(
            project_structure=sorted(all_files),
            main_file_found=main_file,
            working_directory=str(project_dir),
        )

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@app.post("/compile")
async def compile_latex(
    project: UploadFile = File(..., description="ZIP or TAR of LaTeX project"),
    main: Optional[str] = Form(
        None, description="Main TeX file, e.g. main.tex. If omitted, guessed."
    ),
    engine: Literal["xelatex", "lualatex", "pdflatex"] = Form("xelatex"),
    use_biber: Literal["auto", "true", "false"] = Form("auto"),
    timeout_sec: int = Form(300),
    return_type: Literal["pdf", "zip"] = Form("pdf"),
    jobname: Optional[str] = Form(None, description="Optional PDF jobname"),
    force_compile: bool = Form(
        False, description="Force compilation even with missing files"
    ),
):
    """
    Compile a LaTeX project using latexmk, returning a PDF or a ZIP with PDF + logs.
    Enhanced with detailed error reporting and suggestions.
    """
    start_time = datetime.now()
    workdir = Path(tempfile.mkdtemp(prefix="latex-compile-"))

    try:
        logger.info(f"Starting compilation in: {workdir}")

        # Save upload
        archive_path = (
            workdir / f"upload{Path(project.filename or 'project.zip').suffix}"
        )
        with open(archive_path, "wb") as f:
            content = await project.read()
            f.write(content)
        logger.info(f"Saved archive: {archive_path} ({len(content)} bytes)")

        # Extract
        extract_dir = workdir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        _extract_archive(archive_path, extract_dir)

        # Find actual project root
        project_dir = _find_project_root(extract_dir)
        logger.info(f"Project directory: {project_dir}")

        # List files for debugging
        all_files = list(project_dir.rglob("*"))[:20]  # First 20 files
        logger.info(
            f"Project files (sample): {[str(f.relative_to(project_dir)) for f in all_files if f.is_file()]}"
        )

        # Determine main file
        if main:
            main_file = project_dir / main
            if not main_file.exists():
                # Try to find it recursively
                found_files = list(project_dir.rglob(main))
                if found_files:
                    main_file = found_files[0]
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Cannot find specified main file '{main}' in the project.",
                    )
        else:
            main_file = _guess_main_tex(project_dir)
            if not main_file:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot find main .tex file. Please provide 'main' form field or ensure main.tex exists.",
                )

        logger.info(f"Using main file: {main_file}")

        # Build command
        force_biber = use_biber == "true"
        base_cmd = _build_cmd(
            engine=engine,
            force_biber=force_biber,
            main_file=main_file,
            working_dir=project_dir,
        )

        if force_compile:
            base_cmd.insert(-1, "-f")  # Insert -f before the filename

        cmd = base_cmd

        # Run compilation from the directory containing the main file
        working_directory = (
            main_file.parent if main_file.parent != project_dir else project_dir
        )

        code, output = _run(cmd, cwd=working_directory, timeout_sec=timeout_sec)
        execution_time = (datetime.now() - start_time).total_seconds()

        # Find produced PDF
        pdf_path = _find_output_pdf(project_dir, main_file, jobname)

        if not pdf_path:
            raise ValueError("Could not find produced PDF")

        # Gather all log content for analysis
        log_content = output
        log_files = list(project_dir.rglob("*.log"))
        for log_file in log_files[:3]:  # Max 3 log files
            try:
                content = log_file.read_text(errors="ignore")
                log_content += f"\n\n===== {log_file.name} =====\n{content}"
            except Exception as e:
                logger.warning(f"Could not read log file {log_file}: {e}")

        # Look for .blg files (biber logs)
        blg_files = list(project_dir.rglob("*.blg"))
        for blg_file in blg_files[:2]:  # Max 2 blg files
            try:
                content = blg_file.read_text(errors="ignore")
                log_content += f"\n\n===== {blg_file.name} =====\n{content}"
            except Exception as e:
                logger.warning(f"Could not read blg file {blg_file}: {e}")

        # Analyze errors and warnings
        classified_errors = _classify_latex_errors(log_content, output)
        errors = [
            err
            for err in classified_errors
            if err.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.ERROR]
        ]
        warnings = [
            err for err in classified_errors if err.severity == ErrorSeverity.WARNING
        ]

        # Extract compilation statistics
        stats = _extract_compilation_stats(log_content)

        # Check if we have a usable PDF, even if there were warnings/minor errors
        pdf_generated = bool(
            pdf_path and pdf_path.exists() and (pdf_path.stat().st_size > 1000)
        )
        pdf_size = pdf_path.stat().st_size if (pdf_generated and pdf_path) else None

        # Create compilation result
        result = CompilationResult(
            success=code == 0,
            return_code=code,
            pdf_generated=pdf_generated,
            pdf_size=pdf_size,
            errors=errors,
            warnings=warnings,
            execution_time=execution_time,
            pages_count=stats.get("pages"),
        )

        if code == 0 or (
            pdf_generated and code in [1, 12]
        ):  # Accept minor errors if PDF was generated
            success_msg = (
                "Success!"
                if code == 0
                else f"PDF generated with warnings (code {code})"
            )
            logger.info(f"{success_msg} PDF found at: {pdf_path}")

            if return_type == "pdf":
                return StreamingResponse(
                    open(pdf_path, "rb"),
                    media_type="application/pdf",
                    headers={
                        "Content-Disposition": f'inline; filename="{pdf_path.name}"',
                        "X-Compilation-Warnings": str(len(warnings)),
                        "X-Compilation-Time": f"{execution_time:.2f}s",
                        "X-PDF-Pages": str(stats.get("pages", "unknown")),
                    },
                )
            else:
                payload = _zip_with_logs(project_dir, pdf_path, output)
                zip_name = f"{pdf_path.stem}-bundle.zip"
                return StreamingResponse(
                    io.BytesIO(payload),
                    media_type="application/zip",
                    headers={
                        "Content-Disposition": f'attachment; filename="{zip_name}"',
                        "X-Compilation-Warnings": str(len(warnings)),
                        "X-Compilation-Time": f"{execution_time:.2f}s",
                    },
                )
        else:
            # Compilation failed - provide detailed error info
            logger.error("Compilation failed, generating detailed error report")

            # Generate suggestions based on errors
            suggestions = []
            critical_errors = [
                err for err in errors if err.severity == ErrorSeverity.CRITICAL
            ]

            if critical_errors:
                suggestions.append(
                    "🚨 Critical errors found - these must be fixed for compilation to succeed"
                )

            missing_files = [
                err for err in errors if err.type in ["missing_file", "missing_image"]
            ]
            if missing_files:
                suggestions.append(
                    f"📁 {len(missing_files)} missing file(s) detected - check your project structure"
                )

            if len(warnings) > 10:
                suggestions.append(
                    "⚠️ Many warnings detected - consider cleaning up your LaTeX code"
                )

            if code == 124:
                suggestions.append(
                    f"⏱️ Compilation timed out after {timeout_sec}s - try simplifying your document or increasing timeout"
                )

            if not suggestions:
                suggestions.append(
                    "🔍 Check the detailed error messages below for specific issues"
                )

            error_response = ErrorResponse(
                error="Compilation failed",
                details=result,
                timestamp=datetime.now().isoformat(),
                suggestions=suggestions,
            )

            return JSONResponse(status_code=422, content=error_response.model_dump())

    except Exception as e:
        logger.exception("Unexpected error during compilation")
        execution_time = (datetime.now() - start_time).total_seconds()

        error_response = ErrorResponse(
            error=f"Internal error: {str(e)}",
            details=CompilationResult(
                success=False,
                return_code=-1,
                pdf_generated=False,
                execution_time=execution_time,
                errors=[
                    LaTeXError(
                        severity=ErrorSeverity.CRITICAL,
                        type="internal_error",
                        message=str(e),
                        suggestion="This appears to be a server error. Please try again or contact support.",
                    )
                ],
            ),
            timestamp=datetime.now().isoformat(),
            suggestions=["🔧 This appears to be a server error. Please try again."],
        )

        return JSONResponse(status_code=500, content=error_response.model_dump())

    finally:
        # Clean workspace
        shutil.rmtree(workdir, ignore_errors=True)
