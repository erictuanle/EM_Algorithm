function [mu,sigma,pi,classes,nouvelleVraisemblance] = algorithme_EM(X,K,tolerance)
%algorithme_EM permet de faire tourner l'algorithme EM sur les données en
%   entrée
% ARGUMENTS:
%   X données sur lesquels faire tourner l'algorithme EM
%   K: nombre de classes
% SORTIES:
%   mu: matrice de dimension Kxdimension ayant pour ligne les moyennes de chaque distribution
%   sigma: matrices de variance-covariance de chaque distribution, les unes
%       à la suite des autres
%   pi: matrice colonne de probabilité a priori pour chaque ditribution

%% Paramètre
dimension = size(X,2);

%% Phase d'Initialisation
% Probabilité a priori
pi = 1/K*ones(K,1);
% Moyennes
mu = rand(K,dimension);
% Matrice de variance-covariance
sigma = [];
for i=1:K
    sigma = [sigma,eye(dimension)];
end
% Initialisation des vraisemblances
vraisemblance = -Inf;
nouvelleVraisemblance = vraisemblanceLogarithmique(X,mu,sigma,pi);

%% Algorithme Expectation-Maximisation
while abs((nouvelleVraisemblance-vraisemblance)/nouvelleVraisemblance) > tolerance
	vraisemblance = nouvelleVraisemblance;
    % Phase d'Expectation
    T = etape_expectation(X,pi,mu,sigma);
    % Phase de Maximisation
    [mu,sigma,pi] = etape_maximisation(X,T);
    % Calcul de la vraisemblance
    nouvelleVraisemblance = vraisemblanceLogarithmique(X,mu,sigma,pi);
    disp(['Vraisemblance Logarithmique = ' num2str(nouvelleVraisemblance)]);
end

estimated_classes = max(T,[],2);
classes = zeros(size(estimated_classes,1),1);
for i=1:size(estimated_classes,1)
    classes(i) = find(T(i,:)==estimated_classes(i));
end

end

