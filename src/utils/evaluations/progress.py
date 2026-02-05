import sys
from datetime import datetime


class ProgressTracker:
    """Clean, compact progress display with in-place updates."""

    def __init__(self, total_models, total_q1, total_q2, total_step, total_method):
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
        if job_finished:
            self.finished_jobs += 1
            self.current_traj = 0
        self._render()

    def _render(self):
        bar_width = 30
        if self.total_traj > 0:
            total_units = self.total_jobs * self.total_traj
            current_units = (self.finished_jobs * self.total_traj) + self.current_traj
        else:
            total_units = self.total_jobs
            current_units = self.finished_jobs
        progress = current_units / total_units if total_units > 0 else 0
        filled = int(bar_width * progress)
        bar = "#" * filled + "-" * (bar_width - filled)
        current_time = datetime.now().strftime("%H:%M:%S")
        current_info = self.current_model or "NA"
        q_info = f"Q1={self.current_q1} Q2={self.current_q2}"
        method_info = self.current_method or "NA"
        t_delta_info = self.current_t_delta if self.current_t_delta is not None else "NA"
        test_info = f"{q_info} {method_info} tΔ={t_delta_info}"
        phase_info = self.current_phase or "NA"
        dataset_info = self.current_dataset or "NA"

        line1 = f"{phase_info} | {dataset_info}"
        line2 = f"[{bar}] {current_units}/{total_units} | {current_info} | {test_info} | {current_time}"

        if self._has_rendered:
            sys.stdout.write("\033[1A\r\033[K")
        else:
            self._has_rendered = True
        sys.stdout.write(line1 + "\n\r\033[K" + line2)
        sys.stdout.flush()
