
def test_model_trains(processed_data, mock_model):

    mock_model.fit(
        [processed_data[0], processed_data[1][:,:-1,:]],
        processed_data[1][:, 1:, :],
        epochs=1,
        batch_size=2
    )