"""Full settings pane for serving a downloaded model."""

from __future__ import annotations

from dataclasses import dataclass, field

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, Select, Static

# ── Format → runtime mapping ───────────────────────────────────────────

_FORMAT_RUNTIMES: dict[str, list[tuple[str, str]]] = {
    "mlx": [
        ("MLX", "mlx"),
    ],
    "gguf": [
        ("llama.cpp", "llamacpp"),
    ],
    "standard": [
        ("MLX (auto-convert)", "mlx"),
        ("llama.cpp", "llamacpp"),
    ],
    "audio-stt": [
        ("MLX Whisper (STT)", "mlx-audio-stt"),
    ],
    "audio-tts": [
        ("MLX Audio (TTS)", "mlx-audio-tts"),
    ],
    "unknown": [
        ("MLX", "mlx"),
        ("llama.cpp", "llamacpp"),
        ("MLX Whisper (STT)", "mlx-audio-stt"),
        ("MLX Audio (TTS)", "mlx-audio-tts"),
    ],
}

_CONTEXT_SIZES: list[tuple[str, int]] = [
    ("1K (1024)", 1024),
    ("2K (2048)", 2048),
    ("4K (4096)", 4096),
    ("8K (8192)", 8192),
    ("16K (16384)", 16384),
    ("32K (32768)", 32768),
    ("64K (65536)", 65536),
    ("128K (131072)", 131072),
]

_AUDIO_FORMATS: list[tuple[str, str]] = [
    ("WAV", "wav"),
    ("MP3", "mp3"),
    ("Opus", "opus"),
    ("AAC", "aac"),
    ("FLAC", "flac"),
    ("PCM (raw)", "pcm"),
]

_STT_RESPONSE_FORMATS: list[tuple[str, str]] = [
    ("JSON", "json"),
    ("Text", "text"),
    ("Verbose JSON", "verbose_json"),
]


@dataclass(frozen=True)
class ServeSettings:
    """Collected serve settings from the modal."""

    # Server
    name: str
    host: str
    port: int
    runtime: str  # "mlx", "llamacpp", "mlx-audio-stt", "mlx-audio-tts"
    log_level: str  # "warning", "info", "debug"
    restart: str  # "on-failure", "always", "never"
    timeout: int

    # LLM generation defaults
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repeat_penalty: float = 1.0
    stop: list[str] = field(default_factory=list)

    # Context / backend
    context_length: int = 4096
    n_gpu_layers: int = -1

    # STT defaults
    stt_language: str | None = None
    stt_response_format: str = "json"

    # TTS defaults
    tts_voice: str = "default"
    tts_response_format: str = "wav"
    tts_speed: float = 1.0

    # Options
    warmup: bool = False
    stream_by_default: bool = False

    # LiteLLM
    litellm_register: bool = False
    litellm_url: str = ""


def _is_llm(runtime: str) -> bool:
    return runtime in ("mlx", "llamacpp")


def _is_stt(runtime: str) -> bool:
    return runtime == "mlx-audio-stt"


def _is_tts(runtime: str) -> bool:
    return runtime == "mlx-audio-tts"


def _is_audio(runtime: str) -> bool:
    return _is_stt(runtime) or _is_tts(runtime)


class ServeModal(ModalScreen[ServeSettings | None]):
    """Full settings pane before serving a model.

    Adapts options based on model format and selected runtime.
    Dismisses with a ServeSettings dataclass or None on cancel.
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("tab", "focus_next", "Next Field"),
        ("shift+tab", "focus_previous", "Prev Field"),
    ]

    def __init__(
        self,
        repo_id: str,
        default_port: int = 8800,
        model_format: str = "unknown",
        saved: dict | None = None,
    ) -> None:
        super().__init__()
        self._repo_id = repo_id
        self._default_port = default_port
        self._model_format = model_format
        self._saved = saved or {}

        # Load LiteLLM defaults from config
        self._default_litellm_enabled = False
        self._default_litellm_url = ""
        try:
            from silo.config.loader import load_config

            config = load_config()
            self._default_litellm_enabled = config.litellm.enabled
            self._default_litellm_url = config.litellm.url
        except Exception:
            pass

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _s(self, key: str, default):
        """Get a saved value or fall back to default."""
        return self._saved.get(key, default)

    def compose(self) -> ComposeResult:
        s = self._s
        default_name = s(
            "name",
            self._repo_id.split("/")[-1].lower().replace(" ", "-"),
        )
        runtimes = _FORMAT_RUNTIMES.get(
            self._model_format, _FORMAT_RUNTIMES["unknown"]
        )
        default_runtime = s("runtime", runtimes[0][1])
        fallback_ctx = 4096 if self._model_format == "gguf" else 8192
        default_ctx = s("context_length", fallback_ctx)

        with Vertical(id="serve-dialog"):
            yield Static(
                f"[b]Serve Settings[/b]  —  {self._repo_id}  "
                f"[dim]({self._model_format})[/]",
                id="serve-title",
            )

            with VerticalScroll(id="serve-scroll"):
                # ── Server ──
                yield Static("[b dim]Server[/]", classes="serve-section")
                with Horizontal(classes="form-row"):
                    yield Label("Name:")
                    yield Input(value=str(default_name), id="serve-name")
                with Horizontal(classes="form-row"):
                    yield Label("Host:")
                    yield Input(
                        value=str(s("host", "127.0.0.1")), id="serve-host"
                    )
                with Horizontal(classes="form-row"):
                    yield Label("Port:")
                    yield Input(
                        value=str(s("port", self._default_port)),
                        id="serve-port",
                    )
                with Horizontal(classes="form-row"):
                    yield Label("Runtime:")
                    yield Select(
                        runtimes,
                        value=default_runtime,
                        id="serve-runtime",
                    )
                with Horizontal(classes="form-row"):
                    yield Label("Log level:")
                    yield Select(
                        [
                            ("Warning", "warning"),
                            ("Info", "info"),
                            ("Debug", "debug"),
                        ],
                        value=s("log_level", "warning"),
                        id="serve-log-level",
                    )
                with Horizontal(classes="form-row"):
                    yield Label("Restart:")
                    yield Select(
                        [
                            ("On failure", "on-failure"),
                            ("Always", "always"),
                            ("Never", "never"),
                        ],
                        value=s("restart", "on-failure"),
                        id="serve-restart",
                    )
                with Horizontal(classes="form-row"):
                    yield Label("Timeout (s):")
                    yield Input(
                        value=str(s("timeout", 120)), id="serve-timeout"
                    )

                # ── Context (LLM only) ──
                yield Static(
                    "[b dim]Context[/]", classes="serve-section", id="ctx-section"
                )
                with Horizontal(classes="form-row", id="ctx-row"):
                    yield Label("Context:")
                    yield Select(
                        _CONTEXT_SIZES,
                        value=default_ctx,
                        id="serve-context-length",
                    )
                with Horizontal(classes="form-row", id="gpu-layers-row"):
                    yield Label("GPU layers:")
                    yield Input(
                        value=str(s("n_gpu_layers", -1)),
                        id="serve-n-gpu-layers",
                    )
                yield Static(
                    "[dim]-1 = all on GPU, 0 = CPU only[/]",
                    classes="serve-hint",
                    id="gpu-layers-hint",
                )

                # ── LLM Generation Defaults ──
                yield Static(
                    "[b dim]Generation Defaults[/]",
                    classes="serve-section",
                    id="gen-section",
                )
                with Horizontal(classes="form-row", id="gen-max-tokens"):
                    yield Label("Max tokens:")
                    yield Input(
                        value=str(s("max_tokens", 512)),
                        id="serve-max-tokens",
                    )
                with Horizontal(classes="form-row", id="gen-temperature"):
                    yield Label("Temperature:")
                    yield Input(
                        value=str(s("temperature", 0.7)),
                        id="serve-temperature",
                    )
                with Horizontal(classes="form-row", id="gen-top-p"):
                    yield Label("Top P:")
                    yield Input(
                        value=str(s("top_p", 1.0)), id="serve-top-p"
                    )
                with Horizontal(classes="form-row", id="gen-top-k"):
                    yield Label("Top K:")
                    yield Input(
                        value=str(s("top_k", 0)), id="serve-top-k"
                    )
                with Horizontal(classes="form-row", id="gen-min-p"):
                    yield Label("Min P:")
                    yield Input(
                        value=str(s("min_p", 0.0)), id="serve-min-p"
                    )
                with Horizontal(classes="form-row", id="gen-repeat"):
                    yield Label("Repeat penalty:")
                    yield Input(
                        value=str(s("repeat_penalty", 1.0)),
                        id="serve-repeat-penalty",
                    )
                with Horizontal(classes="form-row", id="gen-stop"):
                    yield Label("Stop seqs:")
                    saved_stop = s("stop", [])
                    yield Input(
                        value=", ".join(saved_stop) if saved_stop else "",
                        placeholder="comma-separated, e.g. </s>,<|eot_id|>",
                        id="serve-stop",
                    )

                # ── STT Settings ──
                yield Static(
                    "[b dim]Speech-to-Text[/]",
                    classes="serve-section",
                    id="stt-section",
                )
                with Horizontal(classes="form-row", id="stt-language-row"):
                    yield Label("Language:")
                    yield Input(
                        value=s("stt_language", "") or "",
                        placeholder="auto-detect (or: en, es, fr, de, it, ja, zh...)",
                        id="serve-stt-language",
                    )
                with Horizontal(classes="form-row", id="stt-format-row"):
                    yield Label("Response fmt:")
                    yield Select(
                        _STT_RESPONSE_FORMATS,
                        value=s("stt_response_format", "json"),
                        id="serve-stt-format",
                    )

                # ── TTS Settings ──
                yield Static(
                    "[b dim]Text-to-Speech[/]",
                    classes="serve-section",
                    id="tts-section",
                )
                with Horizontal(classes="form-row", id="tts-voice-row"):
                    yield Label("Voice:")
                    yield Input(
                        value=str(s("tts_voice", "default")),
                        id="serve-tts-voice",
                    )
                with Horizontal(classes="form-row", id="tts-format-row"):
                    yield Label("Audio format:")
                    yield Select(
                        _AUDIO_FORMATS,
                        value=s("tts_response_format", "wav"),
                        id="serve-tts-format",
                    )
                with Horizontal(classes="form-row", id="tts-speed-row"):
                    yield Label("Speed:")
                    yield Input(
                        value=str(s("tts_speed", 1.0)), id="serve-tts-speed"
                    )

                # ── Options ──
                yield Static("[b dim]Options[/]", classes="serve-section")
                with Horizontal(classes="form-row"):
                    yield Label("")
                    yield Checkbox(
                        "Warmup on start (send a test request)",
                        id="serve-warmup",
                        value=s("warmup", False),
                    )
                with Horizontal(classes="form-row", id="stream-default-row"):
                    yield Label("")
                    yield Checkbox(
                        "Stream by default",
                        id="serve-stream-default",
                        value=s("stream_by_default", False),
                    )

                # ── LiteLLM ──
                yield Static(
                    "[b dim]LiteLLM Proxy[/]", classes="serve-section"
                )
                with Horizontal(classes="form-row"):
                    yield Label("")
                    yield Checkbox(
                        "Register with LiteLLM",
                        id="serve-litellm-register",
                        value=s("litellm_register", self._default_litellm_enabled),
                    )
                with Horizontal(classes="form-row", id="litellm-url-row"):
                    yield Label("URL:")
                    yield Input(
                        value=s("litellm_url", self._default_litellm_url),
                        placeholder="e.g. 100.112.188.75 or http://10.0.0.1:4000",
                        id="serve-litellm-url",
                    )

            yield Static(
                "[dim]Tab: next field  Enter: open select / submit  Esc: cancel[/]",
                classes="serve-hint",
            )

            # ── Buttons ──
            with Horizontal(id="buttons"):
                yield Button("Serve", variant="success", id="serve")
                yield Button("Cancel", variant="primary", id="cancel")

    def on_mount(self) -> None:
        self._update_runtime_options()
        # Focus the Serve button so Enter immediately launches
        self.query_one("#serve", Button).focus()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "serve-runtime":
            self._update_runtime_options()

    def _update_runtime_options(self) -> None:
        """Show/hide sections based on selected runtime."""
        runtime = str(self.query_one("#serve-runtime", Select).value)
        is_llm = _is_llm(runtime)
        is_stt = _is_stt(runtime)
        is_tts = _is_tts(runtime)
        is_llamacpp = runtime == "llamacpp"

        # LLM sections
        for wid in (
            "#ctx-section", "#ctx-row",
            "#gen-section",
            "#gen-max-tokens", "#gen-temperature", "#gen-top-p",
            "#gen-top-k", "#gen-min-p", "#gen-repeat", "#gen-stop",
            "#stream-default-row",
        ):
            try:
                self.query_one(wid).display = is_llm
            except Exception:
                pass

        # GPU layers only for llama.cpp
        for wid in ("#gpu-layers-row", "#gpu-layers-hint"):
            try:
                self.query_one(wid).display = is_llamacpp
            except Exception:
                pass

        # STT sections
        for wid in ("#stt-section", "#stt-language-row", "#stt-format-row"):
            try:
                self.query_one(wid).display = is_stt
            except Exception:
                pass

        # TTS sections
        for wid in (
            "#tts-section", "#tts-voice-row",
            "#tts-format-row", "#tts-speed-row",
        ):
            try:
                self.query_one(wid).display = is_tts
            except Exception:
                pass

    # ── Collection helpers ──

    def _int(self, widget_id: str, default: int) -> int:
        val = self.query_one(f"#{widget_id}", Input).value.strip()
        try:
            return int(val) if val else default
        except ValueError:
            return default

    def _float(self, widget_id: str, default: float) -> float:
        val = self.query_one(f"#{widget_id}", Input).value.strip()
        try:
            return float(val) if val else default
        except ValueError:
            return default

    def _collect(self) -> ServeSettings | None:
        """Collect and validate all fields."""
        name = self.query_one("#serve-name", Input).value.strip()
        if not name:
            return None

        host = self.query_one("#serve-host", Input).value.strip() or "127.0.0.1"
        port = self._int("serve-port", 8800)

        runtime = str(self.query_one("#serve-runtime", Select).value)
        log_level = str(self.query_one("#serve-log-level", Select).value)
        restart = str(self.query_one("#serve-restart", Select).value)
        timeout = self._int("serve-timeout", 120)

        ctx_val = self.query_one("#serve-context-length", Select).value
        context_length = int(ctx_val) if ctx_val else 4096

        stop_raw = self.query_one("#serve-stop", Input).value.strip()
        stop = (
            [s.strip() for s in stop_raw.split(",") if s.strip()]
            if stop_raw
            else []
        )

        stt_lang = self.query_one("#serve-stt-language", Input).value.strip() or None
        stt_fmt = str(self.query_one("#serve-stt-format", Select).value)

        tts_voice = self.query_one("#serve-tts-voice", Input).value.strip() or "default"
        tts_fmt = str(self.query_one("#serve-tts-format", Select).value)
        tts_speed = self._float("serve-tts-speed", 1.0)

        litellm_register = self.query_one("#serve-litellm-register", Checkbox).value
        litellm_url_raw = self.query_one("#serve-litellm-url", Input).value.strip()

        # Smart URL normalization — add :4000 if no port provided
        litellm_url = ""
        if litellm_url_raw:
            from silo.litellm.registry import normalize_litellm_url

            litellm_url = normalize_litellm_url(litellm_url_raw)

        return ServeSettings(
            name=name,
            host=host,
            port=port,
            runtime=runtime,
            log_level=log_level,
            restart=restart,
            timeout=timeout,
            max_tokens=self._int("serve-max-tokens", 512),
            temperature=self._float("serve-temperature", 0.7),
            top_p=self._float("serve-top-p", 1.0),
            top_k=self._int("serve-top-k", 0),
            min_p=self._float("serve-min-p", 0.0),
            repeat_penalty=self._float("serve-repeat-penalty", 1.0),
            stop=stop,
            context_length=context_length,
            n_gpu_layers=self._int("serve-n-gpu-layers", -1),
            stt_language=stt_lang,
            stt_response_format=stt_fmt,
            tts_voice=tts_voice,
            tts_response_format=tts_fmt,
            tts_speed=tts_speed,
            warmup=self.query_one("#serve-warmup", Checkbox).value,
            stream_by_default=self.query_one("#serve-stream-default", Checkbox).value,
            litellm_register=litellm_register,
            litellm_url=litellm_url,
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "serve":
            self.dismiss(self._collect())
        else:
            self.dismiss(None)

    def on_key(self, event) -> None:
        """Handle keyboard navigation."""
        focused = self.focused

        # Arrow keys: move between buttons, or between fields
        if event.key in ("left", "right", "up", "down"):
            if isinstance(focused, Button):
                serve_btn = self.query_one("#serve", Button)
                cancel_btn = self.query_one("#cancel", Button)
                if event.key in ("left", "up"):
                    (cancel_btn if focused is serve_btn else serve_btn).focus()
                else:
                    (serve_btn if focused is cancel_btn else cancel_btn).focus()
                event.prevent_default()
                event.stop()
            return

        if event.key != "enter":
            return
        if focused is None:
            return
        # Enter on a Select → toggle it open
        if isinstance(focused, Select):
            focused.action_show_overlay()
            event.prevent_default()
            event.stop()
        # Enter on the Serve button → submit
        elif isinstance(focused, Button) and focused.id == "serve":
            self.dismiss(self._collect())
            event.prevent_default()
            event.stop()
        # Enter on the Cancel button → cancel
        elif isinstance(focused, Button) and focused.id == "cancel":
            self.dismiss(None)
            event.prevent_default()
            event.stop()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Allow Enter from any input to move to the next field."""
        self.action_focus_next()
