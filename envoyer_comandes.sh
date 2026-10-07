#!/bin/bash
# Envoie les modifications sur GitHub
# Utilisation : ./envoyer.sh "Ce que j'ai changé"

git pull
git add .
git commit -m "${1:-Mise à jour}"
git push
