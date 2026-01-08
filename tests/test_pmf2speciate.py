from pmf2speciate import plot_profiles, SourceClassifier
import pytest


@pytest.fixture(scope="module")
def classifier_instance():
    """
    Creates a SourceClassifier instance with the correct path to models.
    """
    # Get the directory of the current test file
    # test_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the models directory relative to the test file
    # models_path = os.path.join(test_dir, "..", "src", "pmf2speciate", "models")
    return SourceClassifier()


def test_plotting():
    """
    Tests the plot_profiles function.
    Note: This test will not check the graphical output, only that the function runs
    without raising an exception.
    """
    try:
        plot_profiles("Combustion", "PM")
    except Exception as e:
        pytest.fail(f"plot_profiles failed with exception: {e}")


def test_source_id(classifier_instance):
    """
    Tests the identify_source function on a sample profile.
    This test uses the 'classifier_instance' fixture to get a pre-configured classifier.
    """
    test_profile = {
        "71-43-2": 2.0199999809265137,
        "50-00-0": 15.479999542236328,
        "74-82-8": 9.380000114440918,
        "109-66-0": 0.2199999988079071,
        "74-98-6": 0.1899999976158142,
        "108-88-3": 0.550000011920929,
        "74-84-0": 0.9100000262260437,
        "106-98-9": 2.059999942779541,
        "115-07-1": 5.440000057220459,
        "67-64-1": 2.4100000858306885,
        "106-99-0": 1.8899999856948853,
        "74-86-2": 4.409999847412109,
        "74-85-1": 18.40999984741211,
        "124-11-8": 0.25999999046325684,
        "111-66-0": 0.30000001192092896,
        "109-67-1": 0.8899999856948853,
        "513-35-9": 0.20999999344348907,
        "75-07-0": 4.829999923706055,
        "590-18-1": 0.5,
        "100-41-4": 0.18000000715255737,
        "124-18-5": 0.4399999976158142,
        "111-84-2": 0.12999999523162842,
        "1120-21-4": 0.5400000214576721,
        "95-47-6": 0.20000000298023224,
        "25339-56-4": 0.5400000214576721,
        "592-41-6": 0.8600000143051147,
        "107-83-5": 0.4099999964237213,
        "107-02-8": 2.380000114440918,
        "100-52-7": 0.5699999928474426,
        "123-72-8": 1.2400000095367432,
        "108-38-3; 106-42-3": 0.30000001192092896,
        "112-40-3": 1.0700000524520874,
        "629-50-5": 0.6700000166893005,
        "91-20-3": 0.6000000238418579,
        "123-38-6": 0.9800000190734863,
        "100-42-5": 0.4099999964237213,
        "107-22-2": 2.180000066757202,
        "66-25-1": 0.2199999988079071,
        "544-76-3": 0.11999999731779099,
        "629-62-9": 0.25999999046325684,
        "629-59-4": 0.5899999737739563,
        "538-68-1": 0.20999999344348907,
        "104-51-8": 0.25999999046325684,
        "108-95-2": 0.25999999046325684,
        "872-05-9": 0.17000000178813934,
        "98-00-0": 2.059999942779541,
    }

    # Use the classifier instance from the fixture
    result = classifier_instance.identify_source(test_profile)

    # Add assertions to verify the functionality
    assert isinstance(result, dict)
    assert "generation_mechanism" in result
    assert "generation_confidence" in result
    assert "specific_source" in result
    assert "source_confidence" in result
    assert "overall_confidence" in result
