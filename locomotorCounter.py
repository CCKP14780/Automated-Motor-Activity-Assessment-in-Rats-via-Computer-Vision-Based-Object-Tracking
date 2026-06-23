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

    def get_supported_grid(self, body_grids):
        """
        body_grids example:
        {
            "head": (1,3),
            "torso": (1,3),
            "tail_base": (1,2)
        }

        Rule:
        - torso must be in the candidate grid
        - total score of body parts in torso's grid >= threshold
        """
        torso_grid = body_grids.get("torso")
        if torso_grid is None:
            return None

        score = 0.0
        supporting_parts = []

        for part, g in body_grids.items():
            if g == torso_grid:
                score += self.body_score[part]
                supporting_parts.append(part)

        if score >= self.threshold:
            return torso_grid

        return None

    def update(self, frame_idx, body_grids):
        supported_grid = self.get_supported_grid(body_grids)

        if supported_grid is None:
            self.candidate_grid = None
            self.candidate_count = 0
            return self.step_count

        if self.committed_grid is None:
            self.committed_grid = supported_grid
            self.log_lines.append(
                f'Frame {frame_idx}: init committed_grid = {self.committed_grid}'
            )
            return self.step_count

        if supported_grid == self.committed_grid:
            self.candidate_grid = None
            self.candidate_count = 0
            return self.step_count

        if supported_grid == self.candidate_grid:
            self.candidate_count += 1
        else:
            self.candidate_grid = supported_grid
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