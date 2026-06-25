# AGAIN Alignment Offset Diagnosis

## Executive Answers

1. Likely cause of the ~3s mismatch: post_roll_or_annotation_end_offset. Visual diagnostics are lightweight heuristics; timing evidence shows annotations start at 0 and usually stop about 3s before video duration.
2. Systematic enough for a global policy: yes; median mismatch is 3.063s.
3. Recommended policy: `drop_last_3s_video_keep_annotation_start`. Fallback: `per_video_end_trim_to_annotation_duration`.
4. Survivors under recommended policy: 993 videos and about 120952 1Hz rows.
5. 1Hz arousal rows can be built safely after documenting the alignment policy: yes.
6. VEATIC-style future spike/change targets can be built after alignment: yes.
7. Small TRIBE encoding pilot: approved only after manual review of the contact sheets confirms the extra seconds are end padding/post-roll; do not scale to all videos yet.
8. First encoding pilot videos: 5F222604-CB86-1ABA-EB7B-31FF78B7BE93_fps_C697A0AC-C316-A3C7-1E20-9767B3E5600D, 772BA0E9-AC30-CEA6-AA1B-18A0300702AA_gallery_AD495E56-6CDA-C100-1557-C6D2922FF843, 8322EDEF-4C09-318A-1B7E-56D62A6839AD_endless_DBD868B7-F648-F70A-2DB2-DF3E78B79A97, 94AA1B20-A5D9-F2B4-FD96-2F20A00FAE89_gallery_A2FD1DBA-2F70-8FF1-5E50-DD8D3E22289C, A135AF2A-692D-9C83-AB54-34F977E4D14D_endless_C090FF31-267B-F7FB-BB2F-D6A4D73B0BE6, B0AAFE06-91C2-2C6D-4547-7508BB719E45_gun_A65112C8-01A8-CCF9-8541-64B569AFF894, B17DF37B-207D-A052-4D45-F862B3BF4C4C_apex_212034DA-3D73-E3F7-7D60-100EDA5FFEAB, B6EE5F90-8F59-D3CF-C449-59216B28207B_fps_B544C913-70BE-6B19-4D48-8B005AB5DE1B, D0F7ABC0-F552-63CD-CAC6-6703C39B83CA_endless_536F0792-E386-C985-C5A4-36DDC461F910, DCB45A10-FAF5-53E0-AB53-2D42AC9BB090_endless_FEF8E949-11A2-A405-E369-EBD7D749EB6D, 0A76B59A-F1FF-F3F5-A3CA-2386FA8EF308_topdown_A6A44F57-D5D1-4F7C-3BFB-411924ED2594, 46A9A47E-EDE3-2EF4-2F06-E95AF67D57E1_topdown_FC1C2B2E-92EF-9F7D-B0E2-553796B3B9CA, B3CE5256-6F65-85EB-0078-689BD5CC1B17_topdown_DB908187-61E5-133A-52C3-5674E44AD52C, BC7EE230-0AF9-3AC5-24E5-8EC4B0B432A8_topdown_C1E0BA23-00AF-FD13-9B00-4F2306FE57EC, DCB45A10-FAF5-53E0-AB53-2D42AC9BB090_gallery_059E4DF3-7153-38D0-3966-D8157C89FCEF, 874A6B13-C999-C802-1138-FA198591D897_solid_ADB2A2F7-134D-65C9-12CC-128AE4A431C4, 99827717-1B21-7ECC-CAD1-99489F7E19EA_solid_446137B1-0CB9-2120-DC76-7541D1FFEE4E, FC50071B-D2BA-C98B-AC6E-7339010D56DA_gallery_8560E34E-106E-978E-68B9-C0F006B32EC6, FC66D29E-8CD8-AE29-A071-33549272C1EE_gun_DCD07A2D-8272-E17B-D277-B7B90C44630E, 647C581E-E72F-0765-F94A-065909B99374_endless_95308EC2-755C-3727-71E1-9460BE0F9C21, 647C581E-E72F-0765-F94A-065909B99374_platform_09E23002-0A95-96C1-2D9D-825462C4C000, AB9921A8-F143-E101-6A47-E66882F571B2_platform_C80B406C-B1A2-6DE9-C6DF-ADADFA7F9EDA, AB9921A8-F143-E101-6A47-E66882F571B2_tiny_3A7D6BD3-8CBE-71EB-1187-627B5E79EEB8, F1430977-52AF-4F46-BF92-0A8D736CBD2B_platform_A01ADF10-31A9-1A25-7048-2EC7A4AD617D.
9. Do not run full TRIBE encoding, do not train models, do not create a final manifest, and do not compare to VEATIC until the alignment convention is frozen.

## Mismatch Distribution

- count: 995
- mean: 3.047s
- median: 3.063s
- std: 0.155s
- min/max: 0.652s / 3.521s
- near +3s: 985
- near -3s: 0
- near 0s: 0

## Visual Diagnostics

- representative videos inspected: 24
- contact sheets: `outputs/again_alignment_offset_diagnosis_20260621_131041/contact_sheets`
- pre-roll evidence votes: 1
- post-roll evidence votes: 7

## Policy Comparison

The recommended policy is global/simple, uses only video and annotation timing, and does not inspect labels for fitting. The per-video end-trim fallback is more exact but should be used only if the global 3s policy leaves too many residual mismatches after manual review.

## Guardrails

tribe_encoding_run=false
models_trained=false
final_manifest_created=false
veatic_outputs_modified=false
