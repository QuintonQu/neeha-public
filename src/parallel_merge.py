from numba import njit, prange
from numba.typed import List
import numpy as np

@njit(parallel=True)
def process_all_lines_x(positive_triggers_t, events_t, events_x_coord, events_y_coord, events_p,
                      total_lines, num_trigger_per_line, mm_per_move, pixel_per_mm,
                      trigger_interval_x, merged_width, merged_height):

    merged = np.zeros((merged_height, merged_width), dtype=np.int8)

    for i in prange(total_lines):
        start_idx = i * num_trigger_per_line
        end_idx = (i + 1) * num_trigger_per_line
        triggers = positive_triggers_t[start_idx:end_idx]

        if triggers.shape[0] < 2:
            continue  # skip empty or incomplete lines

        first_trigger = triggers[0]
        last_trigger = triggers[-1]

        for j in range(events_t.shape[0]):
            t = events_t[j]
            if t <= first_trigger or t >= last_trigger:
                continue

            trig_idx = np.searchsorted(triggers, t)
            if trig_idx == 0 or trig_idx >= triggers.shape[0]:
                continue

            t1 = triggers[trig_idx - 1]
            t2 = triggers[trig_idx]
            ratio = (t - t1) / (t2 - t1 + 1e-8)

            y = int(events_y_coord[j] + pixel_per_mm * i * mm_per_move)
            x = int(((ratio + trig_idx - 1) * trigger_interval_x * pixel_per_mm) + pixel_per_mm * 3 + 1280 - events_x_coord[j])

            # if i % 2 == 0:
            #     val = 1 if events_p[j] == 1 else -1
            # else:
            #     x = merged_width - x - manual_param_x
            #     val = -1 if events_p[j] == 1 else 1
            val = 1 if events_p[j] == 1 else -1

            if 0 <= y < merged_height and 0 <= x < merged_width:
                merged[y, x] += val

    return merged

@njit(parallel=True)
def process_all_lines_y(positive_triggers_t, events_t, events_x_coord, events_y_coord, events_p,
                              total_lines, num_trigger_per_line, mm_per_move, pixel_per_mm,
                              trigger_interval_y, merged_width, merged_height):

    merged = np.zeros((merged_height, merged_width), dtype=np.int32)

    for i in prange(total_lines):
        start = i * num_trigger_per_line
        end = (i + 1) * num_trigger_per_line
        if end > positive_triggers_t.shape[0]:
            continue

        triggers = positive_triggers_t[start:end]
        if triggers.shape[0] < 2:
            continue

        first_trigger = triggers[0]
        last_trigger = triggers[-1]

        for j in range(events_t.shape[0]):
            t = events_t[j]
            if t <= first_trigger or t >= last_trigger:
                continue

            # Binary search (like np.searchsorted)
            trig_idx = 0
            while trig_idx < triggers.shape[0] and triggers[trig_idx] < t:
                trig_idx += 1

            if trig_idx == 0 or trig_idx >= triggers.shape[0]:
                continue

            t1 = triggers[trig_idx - 1]
            t2 = triggers[trig_idx]
            denom = t2 - t1
            if denom == 0:
                continue

            interp = (t - t1) / denom
            y_val = ((interp + trig_idx - 1) * trigger_interval_y * pixel_per_mm) + pixel_per_mm * 3 + events_y_coord[j]
            x_val = (1279 - events_x_coord[j]) + int(pixel_per_mm * i * mm_per_move)

            # if i % 2 == 0:
            #     val = 1 if events_p[j] == 1 else -1
            # else:
            #     y_val = merged_height - y_val - manual_param_y
            #     val = -1 if events_p[j] == 1 else 1

            val = 1 if events_p[j] == 1 else -1

            x_int = int(x_val)
            y_int = int(y_val)

            if 0 <= x_int < merged_width and 0 <= y_int < merged_height:
                merged[y_int, x_int] += val

    return merged

@njit(parallel=True)
def process_line_by_line(positive_triggers_t, events_t, events_x_coord, events_y_coord, events_p,
                      total_lines, num_trigger_per_line, mm_per_move, pixel_per_mm,
                      trigger_interval_x, merged_height, merged_width, manual_shift, axis):

    merged = np.zeros((total_lines, merged_height, merged_width), dtype=np.int8)

    for i in prange(total_lines):
        start_idx = i * num_trigger_per_line
        end_idx = (i + 1) * num_trigger_per_line
        triggers = positive_triggers_t[start_idx:end_idx]

        if triggers.shape[0] < 2:
            continue  # skip empty or incomplete lines

        first_trigger = triggers[0]
        last_trigger = triggers[-1]

        for j in range(events_t.shape[0]):
            t = events_t[j]
            if t <= first_trigger or t >= last_trigger:
                continue

            trig_idx = np.searchsorted(triggers, t)
            if trig_idx == 0 or trig_idx >= triggers.shape[0]:
                continue

            t1 = triggers[trig_idx - 1]
            t2 = triggers[trig_idx]
            ratio = (t - t1) / (t2 - t1 + 1e-8)

            y = int(events_y_coord[j])

            if i % 2 == 0:
                if axis == "y":
                    x = int(((ratio + trig_idx - 1) * trigger_interval_x * pixel_per_mm) + pixel_per_mm * 3 + events_x_coord[j]) # y.raw
                else:
                    x = int(((ratio + trig_idx - 1) * trigger_interval_x * pixel_per_mm) + pixel_per_mm * 3 + 1280 - events_x_coord[j]) # x.raw
                val = 1 if events_p[j] == 1 else -1
            else:
                if axis == "y":
                    x = int(((ratio + trig_idx - 1) * trigger_interval_x * pixel_per_mm) + pixel_per_mm * 3 + 1280 - events_x_coord[j]) # x.raw
                else:
                    x = int(((ratio + trig_idx - 1) * trigger_interval_x * pixel_per_mm) + pixel_per_mm * 3 + events_x_coord[j]) # x.raw
                x = merged_height - x - manual_shift
                val = -1 if events_p[j] == 1 else 1

            merged[i, x, y] += val

    return [merged[i, int(pixel_per_mm * 3):-int(pixel_per_mm * 3), :].copy() for i in range(total_lines)]


@njit(parallel=True)
def process_line_by_line_45deg(positive_triggers_t, events_t, events_x_coord, events_y_coord, events_p,
                      total_lines, num_trigger_per_line, mm_per_move, pixel_per_mm,
                      trigger_interval_x, merged_height, merged_width, manual_shift, axis):

    merged = np.zeros((total_lines, merged_height, merged_width), dtype=np.int16)
    sqrt_2 = np.sqrt(2)

    for i in prange(total_lines):
        start_idx = i * num_trigger_per_line
        end_idx = (i + 1) * num_trigger_per_line
        triggers = positive_triggers_t[start_idx:end_idx]

        if triggers.shape[0] < 2:
            continue  # skip empty or incomplete lines

        first_trigger = triggers[0]
        last_trigger = triggers[-1]

        for j in range(events_t.shape[0]):
            t = events_t[j]
            if t <= first_trigger or t >= last_trigger:
                continue

            trig_idx = np.searchsorted(triggers, t)
            if trig_idx == 0 or trig_idx >= triggers.shape[0]:
                continue

            t1 = triggers[trig_idx - 1]
            t2 = triggers[trig_idx]
            ratio = (t - t1) / (t2 - t1 + 1e-8)
            
            y = int(events_y_coord[j] / sqrt_2 + events_x_coord[j] / sqrt_2)

            if i % 2 == 0:
                x = int(((ratio + trig_idx - 1) * trigger_interval_x * pixel_per_mm) + pixel_per_mm * 3 + events_x_coord[j] / sqrt_2 - events_y_coord[j] / sqrt_2)
                val = 1 if events_p[j] == 1 else -1
            else:
                x = int(((ratio + trig_idx - 1) * trigger_interval_x * pixel_per_mm) + pixel_per_mm * 3 + 1280 - events_x_coord[j] / sqrt_2 + events_y_coord[j] / sqrt_2) 
                x = merged_height - x - manual_shift
                val = -1 if events_p[j] == 1 else 1

            merged[i, x, y] += val

    return [merged[i, int(pixel_per_mm * 3):-int(pixel_per_mm * 3), :].copy() for i in range(total_lines)]