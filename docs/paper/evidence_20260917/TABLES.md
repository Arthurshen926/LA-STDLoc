# 数值附录（2026-09-17）

所有宏平均均为五场景或对应数据集场景等权平均；不是 pooled median。TE 单位 cm，RE 单位度。

## 24 场景主表

| dataset | scene | queries | TE_cm | RE_deg | R5_pct |
| --- | --- | --- | --- | --- | --- |
| 12Scenes | apt1_kitchen | 357 | 0.1598 | 0.0860 | 100.0000 |
| 12Scenes | apt1_living | 493 | 0.1795 | 0.0832 | 100.0000 |
| 12Scenes | apt2_bed | 244 | 0.2441 | 0.1244 | 100.0000 |
| 12Scenes | apt2_kitchen | 230 | 0.1127 | 0.0634 | 99.1304 |
| 12Scenes | apt2_living | 359 | 0.2416 | 0.1004 | 100.0000 |
| 12Scenes | apt2_luke | 624 | 0.2960 | 0.1304 | 99.6795 |
| 12Scenes | office1_gates362 | 386 | 0.2060 | 0.0921 | 100.0000 |
| 12Scenes | office1_gates381 | 1053 | 0.2300 | 0.1095 | 100.0000 |
| 12Scenes | office1_lounge | 327 | 0.2835 | 0.1076 | 98.4709 |
| 12Scenes | office1_manolis | 807 | 0.2260 | 0.1081 | 100.0000 |
| 12Scenes | office2_5a | 497 | 0.5129 | 0.2265 | 96.1771 |
| 12Scenes | office2_5b | 405 | 0.3962 | 0.1409 | 90.3704 |
| 7Scenes | chess | 2000 | 0.2711 | 0.0931 | 99.9000 |
| 7Scenes | fire | 2000 | 0.5246 | 0.2239 | 99.4000 |
| 7Scenes | heads | 1000 | 0.3447 | 0.2184 | 99.6000 |
| 7Scenes | office | 4000 | 0.5970 | 0.1707 | 99.9250 |
| 7Scenes | pumpkin | 2000 | 0.6943 | 0.1651 | 99.6500 |
| 7Scenes | redkitchen | 5000 | 0.4082 | 0.1136 | 99.7000 |
| 7Scenes | stairs | 1000 | 1.5922 | 0.4732 | 91.9000 |
| Cambridge | GreatCourt | 760 | 12.2859 | 0.0620 | 16.0526 |
| Cambridge | KingsCollege | 343 | 17.5562 | 0.1784 | 12.8280 |
| Cambridge | OldHospital | 182 | 8.6311 | 0.1809 | 35.7143 |
| Cambridge | ShopFacade | 103 | 1.8919 | 0.0938 | 85.4369 |
| Cambridge | StMarysChurch | 530 | 3.9914 | 0.1318 | 60.3774 |

## 数据集汇总

| dataset | TE_cm | RE_deg | R5_pct |
| --- | --- | --- | --- |
| Cambridge | 8.8713 | 0.1294 | 42.0818 |
| 7Scenes | 0.6332 | 0.2083 | 98.5821 |
| 12Scenes | 0.2574 | 0.1144 | 98.6524 |

## 原生基线复现

| method | mapping_mask | stage | TE_cm | RE_deg | R5_pct |
| --- | --- | --- | --- | --- | --- |
| STDLoc | off | coarse | 11.2700 | 0.1866 | 28.8329 |
| STDLoc | off | fine | 8.7992 | 0.1309 | 37.9706 |
| STDLoc | on | coarse | 10.8958 | 0.1838 | 28.7873 |
| STDLoc | on | fine | 8.4249 | 0.1345 | 37.5611 |
| ULFLoc | off | coarse | 12.2979 | 0.2062 | 21.6939 |
| ULFLoc | off | fine | 8.4753 | 0.1381 | 40.9189 |
| ULFLoc | on | coarse | 12.3448 | 0.2069 | 21.7233 |
| ULFLoc | on | fine | 8.9853 | 0.1364 | 38.7068 |

## 高斯先验

| prior | TE_cm | RE_deg | R5_pct |
| --- | --- | --- | --- |
| vanilla_2dgs | 9.1311 | 0.1305 | 39.1258 |
| vanilla_3dgs | 8.8092 | 0.1237 | 43.4944 |
| long_lrm_raw | 717.6211 | 11.7265 | 10.4830 |
| long_lrm_1k | 98.7977 | 2.6977 | 22.7898 |
| i3dgs_raw | 23.6528 | 0.4077 | 11.4266 |
| i3dgs_1k | 11.6395 | 0.1847 | 27.3689 |

## 完整组合消融（无筛选删行）

| V2 | fusion | completion | selection | TE_cm | RE_deg | R5_pct | total_anchors | core_mean_ms | loop_fps_scene_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | full | 9.2957 | 0.1328 | 40.6167 | 651753 | 74.9524 | 6.2862 |
| 0 | 0 | 0 | compact | 9.4711 | 0.1385 | 38.7448 | 171067 | 85.7810 | 5.6482 |
| 0 | 0 | 1 | full | 9.3361 | 0.1329 | 40.8696 | 660281 | 75.0626 | 6.1148 |
| 0 | 0 | 1 | compact | 9.3731 | 0.1349 | 38.2577 | 178835 | 82.9133 | 5.6428 |
| 0 | 1 | 0 | full | 8.5668 | 0.1255 | 42.1805 | 651753 | 68.2703 | 6.4157 |
| 0 | 1 | 0 | compact | 9.0065 | 0.1295 | 42.3094 | 171067 | 68.9067 | 6.2003 |
| 0 | 1 | 1 | full | 8.5986 | 0.1240 | 42.1609 | 660281 | 68.2000 | 6.5274 |
| 0 | 1 | 1 | compact | 9.1578 | 0.1313 | 41.8774 | 178835 | 68.2025 | 6.1858 |
| 1 | 0 | 0 | full | 9.0338 | 0.1301 | 39.6655 | 526518 | 72.7694 | 6.2497 |
| 1 | 0 | 0 | compact | 9.3658 | 0.1377 | 38.5382 | 169038 | 81.2521 | 5.5342 |
| 1 | 0 | 1 | full | 8.9759 | 0.1282 | 39.8556 | 535840 | 72.7049 | 6.1047 |
| 1 | 0 | 1 | compact | 9.2461 | 0.1361 | 38.4372 | 176329 | 79.1291 | 6.0627 |
| 1 | 1 | 0 | full | 8.8128 | 0.1268 | 42.3937 | 526518 | 66.6106 | 6.3369 |
| 1 | 1 | 0 | compact | 8.8490 | 0.1297 | 41.9435 | 169038 | 67.7157 | 6.1590 |
| 1 | 1 | 1 | full | 8.7615 | 0.1267 | 42.9955 | 535840 | 66.5209 | 6.3812 |
| 1 | 1 | 1 | compact | 8.8713 | 0.1294 | 42.0818 | 176329 | 67.0995 | 6.1415 |

## 固定共同点几何对照

| geometry | TE_cm | RE_deg | R5_pct |
| --- | --- | --- | --- |
| ray | 8.7654 | 0.1328 | 41.3322 |
| depth | 9.9108 | 0.1597 | 29.1299 |
| center | 11.6673 | 0.1572 | 36.2952 |

## 原生基线预热耗时

| method | mapping_mask | coarse_prefix_ms | coarse_plus_fine_ms |
| --- | --- | --- | --- |
| STDLoc | off | 261.0687 | 403.4214 |
| STDLoc | on | 272.9501 | 423.4511 |
| ULFLoc | off | 570.7524 | 1369.5705 |
| ULFLoc | on | 561.0052 | 1448.4537 |

## 已有 MAtCHA PLY 的转换成本

| scene | minutes | scope |
| --- | --- | --- |
| GreatCourt | 45.9054 | given_MAtCHA_PLY_conversion_only |
| KingsCollege | 47.6603 | given_MAtCHA_PLY_conversion_only |
| OldHospital | 32.4716 | given_MAtCHA_PLY_conversion_only |
| ShopFacade | 9.9042 | given_MAtCHA_PLY_conversion_only |
| StMarysChurch | 38.0090 | given_MAtCHA_PLY_conversion_only |
