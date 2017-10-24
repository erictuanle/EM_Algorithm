function [X, classes] = gaussianmixture(n,pi,mu,sigma)
%gaussianmixture simule un jeu de données issu d'un modèle de mélange
%   gaussien de K composantes en dimension 2
%   INPUTS:
%       n: nombre d'échantillons
%       pi: probabilité a priori des classes
%       mu: moyenne des classes
%       sigma: matrice de covariance des classes
%   OUTPUTS:
%       X: jeu de données cherché
%       classes: association entre les données et leur classe

% Pour connaitre la dimension des échantillons cherchés
dimension = size(mu,2);
% Initialisation
X = zeros(n,dimension);
classes = zeros(n,1);

% Pour tous les échantillons
for i=1:n
    % On choisit au hasard la classe de l'échantillon
    r = rand;
    k = sum(r >= cumsum([0, pi']));
    classes(i) = k;
    % On tire un échantillon aléatoire de la classe choisie au hasard
    echantillon = mvnrnd(mu(k,:),sigma(:,dimension*(k-1)+1:dimension*k));
    % On ajoute l'échantillon au jeu de données
    X(i,:) = echantillon;
end

end

