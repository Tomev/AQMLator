echo "For the tests to run properly use bash run_tests.sh command (WSL on windows)."
echo "The docker image should be called 'alpha'."
# docker run -v ${PWD}:/aqmlator -w /aqmlator alpha python -m unittest
docker run -v ${PWD}:/aqmlator -w /aqmlator alpha tox --current-env
