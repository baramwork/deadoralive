# panic_dip — 독립 감사 (2026-11-01)

감사 범위: `SPEC.md`와 `results/` 산출물만. 구현 코드(`run_panic_dip.py`, `fetch_prices.py`, `tests_panic_dip.py`)는 열지 않은
별도 에이전트가 수행했다(결과물이 자기 자신을 채점하지 않도록). 원문 그대로 보존.

**한 줄 결론:** 산술과 spec 적용은 모두 맞다. 다만 헤드라인 "NO EVIDENCE"의 근거는 portfolio-level 기준 (iii)(iv)(v)이다.
trade-level에서 신호가 없다는 뜻이 아니다. 아래 "신뢰를 낮출 이유" 참조.

## 체크리스트

**1. 원 제안 변형 재계산: PASS** (trades CSV만 사용, net = gross − 0.002)

| 구간 | n | PF 0.10% | PF 0.05% | PF 0.20% | 승률 | 평균 | 중앙 | target/stop/gap/time |
|---|---|---|---|---|---|---|---|---|
| IS | 1577 | 0.4857 | 0.5262 | 0.4143 | 30.37% | −0.851% | −1.20% | 441/760/324/52 |
| OOS | 434 | 0.6925 | 0.7648 | 0.5706 | 32.26% | −0.357% | −1.20% | 116/205/85/28 |

- grid_IS/OOS.csv와 SUMMARY.md 표는 반올림 범위에서 전부 일치. CAGR −18.6%, −5.1%도 총수익에서 재현.
- portfolio 쪽도 trades CSV로 근사 재시뮬레이션(10 slot, 자산의 10%, 같은 날 신호 10 초과 시 선택):
  버려진 거래 수 IS 935 / OOS 12 = grid의 `n_skipped`와 정확히 일치. IS 총수익 −70.8%는 재현 범위 [−75.3%, −61.8%] 안,
  OOS −14.64%는 [−15.3%, −13.0%] 안.

**2. 거래별 산술: PASS** (2011행, 위반 0건) — gross/net 불일치 0, stop 965건 전부 entry×0.99, target 557건 전부 entry×1.03,
gap 409건 전부 entry×0.99 미만(평균 −4.57%, 최악 −42.0%), time 80건 전부 stop~target 사이. exit_date > entry_date 전부.
달력일 간격 1일 1747 / 2일 6 / 3일 248 / 4일 10 (휴장 설명 가능), 5일 초과 없음. 중복·NaN 없음.

**3. 기간 배정: PASS** — IS 진입 2015-11-03~2021-10-29, OOS 2021-11-03~2024-10-31, 범위 밖 0건. HOLDOUT 행 없음,
`holdout_computed`=false. 사소한 경계 누수 둘: OOS 7건이 2024-11-01(HOLDOUT 첫날)에 청산, IS MHK 1건이 2021-11-01에 청산;
extreme-review 목록에 CNC 2025-07-02(HOLDOUT 구간)가 포함 — 성과 계산은 아니지만 §5 "계산조차 하지 않는다" 문구와는 어긋남.

**4. 신호 plausibility: PASS** (일부 unsure) — 진입일 상위: 2020-03-16 (288), 03-12 (224), 03-18 (151), 03-09 (132), 06-11 (55),
03-20 (47), 03-23 (39), 04-01 (38), 03-11 (29), 03-27 (23). 2020-03 한 달이 전체의 48.9%, IS의 62.4%.
−40% 이하 9건: APA/FANG/OXY 2020-03-09(유가 전쟁), PCG 2019-01-14(파산 보도), GL 2024-04-11(short report),
DXCM 2024-07-26(실적 쇼크), EPAM 2022-02-28(우크라이나), CNC 2025-07-02(가이던스 철회, 중간 확신) — 진짜 급락.
NKTR 2018-06-04는 unsure(진입가 788.55는 reverse split 소급 반영으로 보임 — raw 대조 권장).

**5. Grid sanity: PASS** — 72행·조합 유일. DROP별 n_trades 24변형 동일(IS 2814/1577/495, OOS 966/434/120), DROP↑ 엄격 감소.
exit 4종 합 = n_trades, n_filled+n_skipped = n_trades. STOP=none 행은 stop/gap 0. 단조성 위반 없음. `pf_cost005 ≥ pf ≥ pf_cost020`.
기준별 통과 수 재계산 (i)72 (ii)6 (iii)6 (iv)0 (v)14 (vi)7 (vii)11 전부 0 = SUMMARY와 동일. ok_* 플래그 불일치 0.
"IS PF top 10" 표 전 셀 일치. trade-level (i,ii,vi) 통과 1개 = DROP 10% / STOP 없음 / TARGET +5% / HOLD 1.

**6. Placebo: PASS (일관성)** — 144행(72×2; SPEC의 24×2보다 세분, 문제 없음). p95·real_pf가 grid와 완전 일치, beats 플래그 불일치 0.
real PF ≤ placebo p95: IS 50/72, OOS 58/72, 양쪽 47/72. real PF < placebo p05: IS 39개(stop 있는 54개 중 39), OOS 3개.

**7. 기타** — SUMMARY "148개" vs summary.json 147(received vs loaded 기준, 사소). "2873 calendar days"는 trading days.
SPEC §2 "분할 조정 없음"과 달리 Yahoo 가격은 split-adjusted(NKTR 정황) — 수익률엔 무해. SPY OOS MDD 25.4%는 기억상 약 27%와 다름(unsure, 판정 무관).
(viii)에서 실제 trade-level PASS 1 > placebo p95 0이므로 (viii) 단독으론 NO EVIDENCE가 강제되지 않음 — NO EVIDENCE는 "개별 PASS 0"에서 나옴.

## "NO EVIDENCE" 헤드라인의 신뢰를 낮출 이유

1. **portfolio-level 기준이 하루짜리 사건에 지배된다.** 2020-03-06 에너지 10종목이 3/9 시가 gap down 평균 −29.1% → portfolio 하루 −29%.
   원 제안 변형의 FAIL 자체는 견고(2020-03 제외해도 IS PF 0.498; 일 단위 cluster bootstrap 5~95%: IS [0.25, 0.81], OOS [0.56, 0.86]).
   그러나 stop 없는 변형의 판정은 MDD/Sharpe(iii/iv)를 통해 이 한 달에 걸려 있다.
2. **trade-level 잔여 신호가 0이 아니다.** DROP 10% / STOP 없음 / +5% / 1일: IS PF 1.49(placebo p95 0.857), OOS PF 1.23(p95 0.920).
   11개 변형이 두 구간 모두 placebo p95 초과. 탈락 이유는 IS portfolio 총수익 −69.7%, MDD 71.1%, Sharpe −0.85.
   IS에서 PF > 1인 17개 변형 모두 portfolio 수익 음수 — 원인은 slot cap(642/1577 체결), "낙폭 큰 순" 선택, 2020-03 100% 투입.
   정확한 표현: "신호가 없다"가 아니라 **"SPEC §4 자금관리 아래 거래 가능한 edge가 없다"**.
3. **placebo가 변동성·시기·군집을 맞추지 않았다.** 무작위 (날짜, 종목)은 급락 직후 종목보다 변동성이 훨씬 낮다. tight stop은 고변동성을 부당하게
   벌주고, stop 없는 target 청산은 고변동성에 유리. "beats placebo" 11개와 "placebo 미만" 50/72 모두 신호 효과와 변동성 효과가 섞임.
4. **커버리지 손실이 133개보다 크다.** 1~15행 stub 8개(AVB, EA, EQR, LEG, QRVO, WBD, SATS, PSKY) 포함 최소 141/751(≈18.8%).
   404 상당수는 ticker 개명(ANTM→ELV, ABC→COR, FLT→CPAY, RE→EG, TMK→GL, UTX→RTX 등). 진짜 상폐·인수·파산(SIVB, FRC, CHK, ENDP, FTR, CTL 등)
   누락은 dip-buy 성과를 부풀리는 방향 — FAIL은 더 믿을 만하고 PASS(특히 stop 없는 변형)는 더 의심.
5. **유효 표본이 작다.** IS 1577건은 361 거래일·69개월에 흩어져 있고 62%가 한 달. "거래수 ≥ 100"은 독립 표본 수가 아님.
6. **체결 가정이 낙관적일 수 있다.** 종가 체결, target은 high 터치만으로 체결, 편도 0.10% 고정. gap 청산은 시가라 현실적.
   0.20% 민감도에서도 원 제안 PF 0.41 — 구제되지 않음.
7. **사전등록 시점은 감사자가 git 없이 검증 못 함.** (운영자 주: git 이력상 SPEC 커밋 → 구현 커밋 순서이며, §6 (viii) 문구 정정은
   SPEC 첫 커밋에 포함되어 있음.)
