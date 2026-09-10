# OAI AllClinical00 Feature Map

The model treats `P01XRKOA` as its independent baseline outcome. OAI defines the code as radiographic knee OA status by person:

- `0`: neither knee
- `1`: right knee only
- `2`: left knee only
- `3`: both knees

The preparation script converts `1`, `2`, and `3` to `knee_oa_confirmed=yes`, then excludes every radiographic source field from model inputs.

| Training feature | OAI source | Meaning |
| --- | --- | --- |
| `age_years` | `V00AGE` | Baseline age |
| `bmi` | `P01BMI` | Calculated BMI |
| `frequent_knee_symptom_status` | `P01KSX` | Person-level frequent knee-pain status |
| `prior_knee_injury` | `P02KINJ` | Prior knee injury with walking difficulty |
| `family_history_knee_replacement` | `P02FAMHXKR` | Blood relative with knee replacement |
| `frequent_stair_climbing` | `P02PA1` | At least 10 flights on most days |
| `frequent_kneeling` | `P02PA2` | Kneeling for 30+ minutes on most days |
| `frequent_deep_knee_bending` | `P02PA3` | Deep knee bending for 30+ minutes on most days |
| `womac_total_worse_knee` | `V00WOMTSL`, `V00WOMTSR` | Larger side-specific WOMAC total |
| `koos_pain_worse_knee` | `V00KOOSKPR`, `V00KOOSKPL` | Lower side-specific KOOS pain score |
| `patellofemoral_crepitus_any` | `V00RKPFCRE`, `V00LKPFCRE` | Crepitus on either knee exam |
| `knee_effusion_any` | `V00RKEFFB`, `V00LKEFFB` | Effusion on either knee exam |
| `flexion_contracture_worse_deg` | `V00RKFHDEG`, `V00LKFHDEG` | Largest non-negative flexion contracture |
| `chair_stand_time_sec_mean` | `V00CSTIME1`, `V00CSTIME2` | Mean chair-stand trial time |
| `walk_400m_time_sec` | `V00400MTIM` | 400-metre walk time |

This is a person-level model for OA in either knee. Its percentages are research screening estimates only. OAI is a US research cohort, so performance must be re-evaluated before applying the model to a North Eastern Indian population or using it in clinical care.
