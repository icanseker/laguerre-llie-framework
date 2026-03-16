function [enhanced, best_nu, best_t, total_evals] = run_laguerre(img_low, preset)
%RUN_LAGUERRE  Enhance a low-light image using Laguerre polynomial coefficient bounds.
%
%  Paper: "An Adaptive and Scalable Convolution Framework for Low-Light Image Enhancement via Analytic Functions Subordinate to Laguerre Polynomials"
%
%  ── Architecture ────────────────────────────────────────────────────────────
%  Pure function: image in → enhanced image out (plus diagnostics).
%  All optimization parameters — Phase 1 grid density AND Phase 2 Nelder-Mead
%  tolerances — are resolved internally from the single 'preset' string.
%
%  ── Core computational optimization: spatial precomputation ─────────────────
%  The Master Kernel decomposes as:
%
%      I * K_master  =  a₂ · img_d  +  (1+a₃)/8 · I_edge
%
%  where:
%      img_d  = double(img_low)        — identity component, computed once
%      I_edge = img_low * H_edge       — 8-neighbor sum, computed once
%      H_edge = [1,1,1; 1,0,1; 1,1,1]
%
%  This decomposition is an exact algebraic identity (not an approximation).
%  It moves the heavy spatial convolution entirely outside the optimization
%  loop. Each of the ~N_evals entropy evaluations reduces to:
%      - 2 scalar-matrix multiplications
%      - 1 element-wise addition
%      - entropy calculation
%  No conv2 call inside the loop.
%
%  ── Complexity reduction ─────────────────────────────────────────────────────
%  Complexity: O(W×H×K²) + O(N_evals × W×H)
%             — spatial cost isolated from parametric search
%
%  The freed budget is used to search a much finer grid at the same wall-clock
%  time. high_precision uses ~663 grid points, covering the [0.51,0.99] × [-1.0,0.9]
%  parameter space densely. Both ν bounds strictly open: ν = 0.5 → (1−ν) = 0.5 (boundary); ν = 1.0 → (1−ν) = 0 (degenerate).
%
%  ── Preset system ────────────────────────────────────────────────────────────
%  Both Phase 1 (grid density) and Phase 2 (Nelder-Mead tolerances) scale
%  together. The freed computation budget is used for finer grids, not fewer
%  evaluations — the same number of evaluations now achieves higher parameter
%  resolution. Preset differentiation remains meaningful: more grid points =
%  higher probability of finding the globally optimal (ν*, t*).
%
%  'high_precision'  Phase 1: nu_step=0.03, t_step=0.05  → ~663 grid evals
%                    Phase 2: n_cand=5, TolFun=1e-6, TolX=1e-6, MaxFunEvals=1100
%
%  'balanced'        Phase 1: nu_step=0.05, t_step=0.08  → ~240 grid evals
%                    Phase 2: n_cand=3, TolFun=1e-4, TolX=1e-4, MaxFunEvals=400
%
%  'fast'            Phase 1: nu_step=0.10, t_step=0.20  → ~50 grid evals
%                    Phase 2: n_cand=1, TolFun=1e-2, TolX=1e-2, MaxFunEvals=100
%
%  ── Inputs ──────────────────────────────────────────────────────────────────
%  img_low  - Low-light color image (H × W × 3, uint8)
%             Must have I_mean < 64 (enforced by caller in main.m)
%  preset   - Precision tier: 'high_precision' | 'balanced' | 'fast'
%
%  ── Outputs ─────────────────────────────────────────────────────────────────
%  enhanced    - Enhanced image (H × W × 3, uint8)
%  best_nu     - Optimal ν found by two-phase optimization
%  best_t      - Optimal t found by two-phase optimization
%  total_evals - Total entropy evaluations (grid + all Nelder-Mead runs)

    % ── Parameter domain ──────────────────────────────────────────────────────
    NU_MIN = 0.51;  NU_MAX = 0.99;
    T_MIN  = -1.0;  T_MAX  =  0.9;

    % ── Resolve ALL optimization parameters from preset ────────────────────────
    % Finer grids are now affordable because the precomputation removes
    % conv2 from the inner loop. The same wall-clock budget covers 6× more
    % parameter space, improving global optimum discovery.
    switch lower(preset)
        case 'high_precision'
            % Finest grid: ~663 points across [0.51,0.99] × [-1.0,0.9]
            % Every 0.03 step in ν, every 0.05 step in t
            nu_step        = 0.03;  % 17 ν values
            t_step         = 0.05;   % 39  t values  → ~663 grid points
            n_candidates   = 6;
            nm_TolFun      = 1e-6;
            nm_TolX        = 1e-6;
            nm_MaxFunEvals = 1100;
        case 'balanced'
            % Medium grid: ~240 points
            nu_step        = 0.05;  % 10 ν values
            t_step         = 0.08;   % 24 t values  → ~240 grid points
            n_candidates   = 4;
            nm_TolFun      = 1e-4;
            nm_TolX        = 1e-4;
            nm_MaxFunEvals = 400;
        case 'fast'
            % Coarse grid: ~50 points
            nu_step        = 0.10;  % 5 ν values
            t_step         = 0.20;   % 10 t values  → ~50 grid points
            n_candidates   = 2;
            nm_TolFun      = 1e-2;
            nm_TolX        = 1e-2;
            nm_MaxFunEvals = 100;
        otherwise
            error('run_laguerre:unknownPreset', ...
                  'Unknown preset "%s". Valid: high_precision | balanced | fast', preset);
    end


    %% ═══════════════════════════════════════════════════════════════════════
    %  MATHEMATICAL DECOUPLING — Spatial Precomputation
    %
    %  The Master Kernel separates into two structurally distinct components:
    %
    %      K_master = a₂ · δ  +  e · H_edge
    %
    %  where δ is the identity filter (Kronecker delta) and H_edge is the
    %  8-neighbor ring mask — both are STATIC (parameter-independent):
    %
    %      δ      = [0,0,0; 0,1,0; 0,0,0]   (center pass-through)
    %      H_edge = [1,1,1; 1,0,1; 1,1,1]   (8-neighbor uniform sum)
    %      e      = (1 + a₃) / 8             (scalar, changes per eval)
    %
    %  By the distributive property of convolution:
    %
    %      I ∗ K_master = I ∗ (a₂·δ + e·H_edge)
    %                   = a₂·(I ∗ δ) + e·(I ∗ H_edge)
    %                   = a₂·img_d   + e·I_edge        [since I ∗ δ = I]
    %
    %  I_edge = I ∗ H_edge depends ONLY on the image — not on (ν, t).
    %  It is computed once here, before all optimization loops.
    %
    %  Complexity reduction:
    %  Complexity: O(W×H×K²) + O(N_evals × W×H)
    %              — spatial K² cost isolated from parametric search
    %
    %  The freed compute budget is reinvested into a finer parameter grid
    %  (6× more points in high_precision), increasing the probability of
    %  seeding Nelder-Mead near the true global entropy maximum.
    %  ═══════════════════════════════════════════════════════════════════════

    img_d  = double(img_low);
    H_edge = [1, 1, 1;
              1, 0, 1;
              1, 1, 1];

    % Precompute 8-neighbor sum for each channel
    [rows, cols, nch] = size(img_d);
    I_edge = zeros(rows, cols, nch);
    for ch = 1:nch
        I_edge(:,:,ch) = imfilter(img_d(:,:,ch), H_edge, 'replicate', 'same');
    end


    %% ═══════════════════════════════════════════════════════════════════════
    %  PHASE 1: COARSE GRID SEARCH
    %
    %  Each evaluation uses the precomputed (img_d, I_edge) — no conv2.
    %  With precomputation, the finer grid (nu_step=0.015) explores the
    %  parameter space more densely at the same computational cost.
    %  ═══════════════════════════════════════════════════════════════════════

    nu_range = NU_MIN:nu_step:NU_MAX;
    t_range  = T_MIN:t_step:T_MAX;
    if nu_range(end) < NU_MAX, nu_range = [nu_range, NU_MAX]; end
    if t_range(end)  < T_MAX,  t_range  = [t_range,  T_MAX];  end

    max_grid = length(nu_range) * length(t_range);
    results  = zeros(max_grid, 3);
    count    = 0;

    for nu = nu_range
        for t = t_range
            [a2, a3, valid] = laguerre_coefficients(nu, t);
            if ~valid, continue; end

            % Fast evaluation — no conv2, uses precomputed spatial components
            edge_factor = (1 + a3) / 8;
            acc = a2 * img_d + edge_factor * I_edge + 50;
            candidate = uint8(min(max(round(acc), 0), 255));
            ent = calc_entropy(candidate);

            count = count + 1;
            results(count, :) = [nu, t, ent];
        end
    end

    results     = results(1:count, :);
    total_evals = count;

    [~, si] = sort(results(:,3), 'descend');
    results  = results(si, :);

    n_cand = min(n_candidates, count);


    %% ═══════════════════════════════════════════════════════════════════════
    %  PHASE 2: NELDER-MEAD SIMPLEX REFINEMENT
    %
    %  Uses fast_neg_entropy.m which accepts precomputed (img_d, I_edge).
    %  No conv2 calls inside Nelder-Mead either.
    %  ═══════════════════════════════════════════════════════════════════════

    nm_opts = optimset( ...
        'TolX',        nm_TolX,        ...
        'TolFun',      nm_TolFun,      ...
        'MaxIter',     300,             ...
        'MaxFunEvals', nm_MaxFunEvals,  ...
        'Display',     'off'            ...
    );

    best_entropy = -Inf;
    best_nu = results(1, 1);
    best_t  = results(1, 2);

    for k = 1:n_cand
        % Objective uses precomputed spatial components — no conv2
        obj = @(x) fast_neg_entropy(x, img_d, I_edge, @laguerre_coefficients, ...
                                    NU_MIN, NU_MAX, T_MIN, T_MAX);

        [x_opt, fval, ~, output] = fminsearch(obj, [results(k,1), results(k,2)], nm_opts);
        total_evals = total_evals + output.funcCount;

        x_opt(1) = max(NU_MIN, min(NU_MAX, x_opt(1)));
        x_opt(2) = max(T_MIN,  min(T_MAX,  x_opt(2)));

        if -fval > best_entropy
            best_entropy = -fval;
            best_nu = x_opt(1);
            best_t  = x_opt(2);
        end
    end


    %% ═══════════════════════════════════════════════════════════════════════
    %  PRODUCE FINAL ENHANCED IMAGE
    %
    %  apply_convolution.m is called exactly once with the optimal (ν*, t*).
    %  This keeps the full documented pipeline for reproducibility and ensures
    %  the final output is identical to what apply_convolution would produce
    %  for any direct call outside the optimization context.
    %  ═══════════════════════════════════════════════════════════════════════

    [a2, a3, ~] = laguerre_coefficients(best_nu, best_t);
    enhanced = apply_convolution(img_low, a2, a3);

end


%% ═══════════════════════════════════════════════════════════════════════════
%  LOCAL FUNCTION: laguerre_coefficients
%  ═══════════════════════════════════════════════════════════════════════════
function [a2, a3, valid] = laguerre_coefficients(nu, t)
%LAGUERRE_COEFFICIENTS  Compute Laguerre polynomial coefficient bounds.
%
%  Formulas (Theorem 1, Laguerre family):
%
%    a₁ = 1  (fixed by bi-univalent normalization)
%
%    numer = 2·(1−ν)³
%    denom = (1−t)·(ν² − t·ν·(2−ν) + 2(1−ν))
%    a₂ = sqrt(numer / denom)
%
%    a₃ = (1−ν)/[(1−t)(t+2)]  +  (1−ν)²/(1−t)²
%
%  Domain: ν ∈ (0.5, 1.0), t ∈ [−1, 1).
%  Both ν bounds strictly open: ν=0.5 boundary, ν=1.0 makes (1−ν)=0.

    valid = false; a2 = 0; a3 = 0;

    numer = 2 * (1 - nu)^3;
    denom = (1 - t) * (nu^2 - t*nu*(2 - nu) + 2*(1 - nu));
    if denom <= 0 || numer < 0, return; end
    a2 = sqrt(numer / denom);

    td1 = (1 - t) * (t + 2);
    td2 = (1 - t)^2;
    if td1 == 0 || td2 == 0, return; end
    a3 = (1 - nu) / td1 + (1 - nu)^2 / td2;

    if a2 <= 0 || a3 <= 0 || ~isfinite(a2) || ~isfinite(a3), return; end
    valid = true;
end