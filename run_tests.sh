echo "For the tests to run properly use bash run_tests.sh command (WSL on windows)."
echo "The docker image should be called 'aqmlator:alpha'."
docker run -v ${PWD}:/aqmlator -w /aqmlator aqmlator:alpha tox --current-env
