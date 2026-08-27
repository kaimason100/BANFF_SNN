%% Batch launcher for the breast-cancer recurrent-regime search
% For an inline interactive report, open the same-named .mlx Live Script.

if ~exist('banff_regime_search_settings', 'var')
    banff_regime_search_settings = struct();
end
search_report = banff_search_breast_cancer_recurrent_regime( ...
    banff_regime_search_settings);
