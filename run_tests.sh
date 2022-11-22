echo "For the tests to run properly use bash run_tests.sh command (from PS on windows)."
echo "The docker image should be called 'alpha'."
docker run -v ${PWD}:/aqmlator -w /aqmlator alpha python -m unittest
