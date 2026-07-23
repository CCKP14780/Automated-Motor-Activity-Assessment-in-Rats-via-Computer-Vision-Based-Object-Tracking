class LocomotorCounter:
    def __init__(self, track_id, body_score, threshold=2.5, confirm_frames=3):
        self.track_id = track_id
        self.body_score = body_score
        self.threshold = threshold
        self.confirm_frames = confirm_frames

        self.committed_grid = None
        self.candidate_grid = None
        self.candidate_count = 0
        self.step_count = 0

        self.log_lines = []

    def get_current_grid(self, body_grids):
        """
        Determine the rat's current grid using the
        weighted multi-markpoint algorithm.

        Rules:
        - torso must exist
        - sum of body-part weights in torso's grid
        must exceed threshold

        Return the grid occupied by the selected body part.
        Returns None if that body part is not detected.
        """

        torso_grid = body_grids.get("torso")

        if torso_grid is None:
            return None

        score = 0.0

        for part, grid in body_grids.items():
            if grid == torso_grid:
                score += self.body_score[part]

        if score >= self.threshold:
            return torso_grid

        return None

    def update(self, frame_idx, body_grids):
        current_grid = self.get_current_grid(body_grids)

        if current_grid is None:
            self.candidate_grid = None
            self.candidate_count = 0
            return self.step_count

        if self.committed_grid is None:
            self.committed_grid = current_grid
            self.log_lines.append(
                f'Frame {frame_idx}: init committed_grid = {self.committed_grid}'
            )
            return self.step_count

        if current_grid == self.committed_grid:
            self.candidate_grid = None
            self.candidate_count = 0
            return self.step_count

        if current_grid == self.candidate_grid:
            self.candidate_count += 1
        else:
            self.candidate_grid = current_grid
            self.candidate_count = 1

        if self.candidate_count >= self.confirm_frames:
            old_grid = self.committed_grid
            new_grid = self.candidate_grid

            self.step_count += 1
            self.committed_grid = new_grid

            self.log_lines.append(
                f'Frame {frame_idx}: {old_grid} -> {new_grid} | Steps = {self.step_count}'
            )

            self.candidate_grid = None
            self.candidate_count = 0

        return self.step_count

    def get_log(self):
        return "\n".join(self.log_lines)
    
class SingleMarkpointCounter(LocomotorCounter):
    """
    Determine the rat position using only one body part.
    """

    def __init__(
        self,
        track_id,
        body_part,
        confirm_frames=3
    ):
        super().__init__(
            track_id=track_id,
            body_score={},
            threshold=0,
            confirm_frames=confirm_frames
        )

        self.body_part = body_part

    def get_current_grid(self, body_grids):
        return body_grids.get(self.body_part)