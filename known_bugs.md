## changing self.maximum and self.minimum values
reducing self.maximum by too much would cause the pipeline to crash with size mismatch error since `sample_points_and_speed_from_pos_neural` assumes more than 5000 eligible points.

## occupancy grid returns no possible_locs
fixed by searching for a larger area for candidates.