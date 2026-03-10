import logging
import shutil
import sys
import threading
from datetime import datetime


class ProgressTracker:
    """Compact single-line progress display with in-place updates."""

    _active_tracker = None
    _lock = threading.RLock()

    def __init__(
        self,
        total_models,
        total_q1,
        total_q2,
        total_step,
        total_method,
        unit_offset: int = 0,
        global_total_units: int | None = None,
    ):
        self.total_models = total_models
        self.total_q1 = total_q1
        self.total_q2 = total_q2
        self.total_step = total_step
        self.total_method = total_method
        self.total_jobs = total_models * total_q1 * total_q2 * total_step * total_method
        self.finished_jobs = 0
        self.current_model = ""
        self.current_model_idx = 0
        self.current_q1 = 0
        self.current_q2 = 0
        self.current_q1_idx = 0
        self.current_q2_idx = 0
        self.current_step_idx = 0
        self.current_method_idx = 0
        self.current_method = ""
        self.current_t_delta = None
        self.current_traj = 0
        self.total_traj = 0
        self.total_units = 0
        self.current_dataset = ""
        self.current_phase = ""
        self._has_rendered = False
        self._last_line = ""
        self.unit_offset = max(0, int(unit_offset))
        self.global_total_units = (
            max(0, int(global_total_units))
            if global_total_units is not None
            else None
        )

    def update(
        self,
        model=None, model_idx=None,
        q1=None, q2=None,
        q1_idx=None, q2_idx=None,
        step_idx=None, method_idx=None,
        method=None, t_delta=None,
        dataset=None, phase=None,
        traj=None, total_traj=None,
        job_finished=False
    ):
        """Update progress display."""
        if model is not None:
            self.current_model = model
        if model_idx is not None:
            self.current_model_idx = model_idx + 1  # 1-indexed
        if q1 is not None:
            self.current_q1 = q1
        if q2 is not None:
            self.current_q2 = q2
        if q1_idx is not None:
            self.current_q1_idx = q1_idx + 1
        if q2_idx is not None:
            self.current_q2_idx = q2_idx + 1
        if step_idx is not None:
            self.current_step_idx = step_idx + 1
        if method_idx is not None:
            self.current_method_idx = method_idx + 1
        if method is not None:
            self.current_method = method
        if t_delta is not None:
            self.current_t_delta = t_delta
        if dataset is not None:
            self.current_dataset = dataset
        if phase is not None:
            self.current_phase = phase
        if traj is not None:
            self.current_traj = traj
        if total_traj is not None:
            self.total_traj = total_traj
        terminal_job = False
        if job_finished:
            self.finished_jobs += 1
            self.current_traj = 0
            terminal_job = self.total_jobs > 0 and self.finished_jobs >= self.total_jobs
        if terminal_job:
            self._deactivate()
            return
        self._render()

    @staticmethod
    def _truncate(text: str, width: int) -> str:
        if width <= 0:
            return ""
        if len(text) <= width:
            return text
        if width <= 3:
            return text[:width]
        return text[: width - 3] + "..."

    def _current_and_total_units(self) -> tuple[int, int]:
        if self.total_traj > 0:
            local_total_units = self.total_jobs * self.total_traj
            local_current_units = (self.finished_jobs * self.total_traj) + self.current_traj
        else:
            local_total_units = self.total_jobs
            local_current_units = self.finished_jobs
        if self.global_total_units is not None and self.global_total_units > 0:
            total_units = self.global_total_units
            current_units = min(self.global_total_units, self.unit_offset + local_current_units)
        else:
            total_units = local_total_units
            current_units = local_current_units
        return int(current_units), int(total_units)

    def _build_line(self) -> str:
        bar_width = 30
        current_units, total_units = self._current_and_total_units()
        progress = current_units / total_units if total_units > 0 else 0
        filled = int(bar_width * progress)
        bar = "#" * filled + "-" * (bar_width - filled)
        current_time = datetime.now().strftime("%H:%M:%S")
        current_info = self.current_model or "NA"
        q_info = f"Q1={self.current_q1} Q2={self.current_q2}"
        method_info = self.current_method or "NA"
        t_delta_info = self.current_t_delta if self.current_t_delta is not None else "NA"
        traj_info = (
            f"traj={self.current_traj}/{self.total_traj}"
            if self.total_traj > 0
            else "traj=NA"
        )
        test_info = f"{q_info} {method_info} tΔ={t_delta_info} {traj_info}"
        phase_info = self.current_phase or "NA"
        dataset_info = self.current_dataset or "NA"
        progress_pct = progress * 100.0
        line = (
            f"{phase_info} | {dataset_info} | "
            f"[{bar}] {current_units}/{total_units} ({progress_pct:5.1f}%) | "
            f"{current_info} | {test_info} | {current_time}"
        )
        term_width = shutil.get_terminal_size(fallback=(160, 20)).columns
        return self._truncate(line, max(40, term_width - 1))

    def _clear_render_locked(self) -> None:
        if not self._has_rendered:
            return
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def _write_render_locked(self) -> None:
        sys.stdout.write("\r\033[K" + self._last_line)
        sys.stdout.flush()

    def _finalize(self) -> None:
        with self._lock:
            if not self._has_rendered:
                return
            sys.stdout.write("\r\033[K" + self._last_line + "\n")
            sys.stdout.flush()
            self._has_rendered = False
            if type(self)._active_tracker is self:
                type(self)._active_tracker = None

    def _deactivate(self) -> None:
        with self._lock:
            if not self._has_rendered:
                return
            self._clear_render_locked()
            self._has_rendered = False
            if type(self)._active_tracker is self:
                type(self)._active_tracker = None

    def _render(self):
        self._last_line = self._build_line()
        with self._lock:
            type(self)._active_tracker = self
            if not self._has_rendered:
                self._has_rendered = True
            self._write_render_locked()

    @classmethod
    def _emit_log_message(cls, stream, message: str) -> None:
        with cls._lock:
            active = cls._active_tracker
            if active is not None:
                active._clear_render_locked()
            stream.write(message + "\n")
            stream.flush()
            if active is not None and active._has_rendered:
                active._write_render_locked()

    @classmethod
    def clear_active(cls, *, final: bool = False) -> None:
        with cls._lock:
            active = cls._active_tracker
            if active is None:
                return
            active._clear_render_locked()
            if final:
                sys.stdout.write("\n")
                sys.stdout.flush()
                active._has_rendered = False
                cls._active_tracker = None


class ProgressAwareStreamHandler(logging.StreamHandler):
    """Logging handler that preserves in-place progress rendering."""

    def emit(self, record) -> None:
        try:
            msg = self.format(record)
            ProgressTracker._emit_log_message(self.stream, msg)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)
