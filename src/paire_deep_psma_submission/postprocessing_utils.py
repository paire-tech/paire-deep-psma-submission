import SimpleITK as sitk


def post_process_FDG_based_on_PSMA_classes(
    preds_fdg_sitk: sitk.Image,  # binary mask (FDG lesions)
    totalsegmentator_resampled_to_fdg: sitk.Image,  # segmentation of organs/classes
    preds_psma_sitk: sitk.Image,  # binary mask (PSMA lesions)
    totalsegmentator_resampled_to_psma: sitk.Image,  # segmentation of organs/classes
) -> sitk.Image:
    """
    Remove FDG lesions (connected components) if no corresponding lesion is present
    in PSMA for the same anatomical class (based on TotalSegmentator labels).
    """

    # Label connected components in FDG predictions
    fdg_cc = sitk.ConnectedComponent(preds_fdg_sitk)
    num_labels = int(sitk.GetArrayViewFromImage(fdg_cc).max())

    # Convert everything to numpy for efficient processing
    fdg_cc_np = sitk.GetArrayFromImage(fdg_cc)
    ts_fdg_np = sitk.GetArrayFromImage(totalsegmentator_resampled_to_fdg)
    psma_np = sitk.GetArrayFromImage(preds_psma_sitk) > 0
    ts_psma_np = sitk.GetArrayFromImage(totalsegmentator_resampled_to_psma)

    # Find all classes with PSMA lesions
    psma_classes = set(ts_psma_np[psma_np])

    # Create an empty array to store post-processed FDG mask
    fdg_post_np = np.zeros_like(fdg_cc_np, dtype=np.uint8)

    for label_id in range(1, num_labels + 1):
        lesion_mask = fdg_cc_np == label_id
        if not lesion_mask.any():
            continue

        # Get class labels from TotalSegmentator for this lesion
        class_labels = ts_fdg_np[lesion_mask]
        unique_classes = np.unique(class_labels)

        kept = any(c in psma_classes for c in unique_classes if c != 0)
        # the idea is to remove lesions that are only in one total segmentators classes that do not match any totalsegmentator of psma

        if kept:
            fdg_post_np[lesion_mask] = 1  # keep lesion
        # else:
        # print(f"removed lesion {label_id} of class{unique_classes}")

    # Convert back to SimpleITK image
    preds_fdg_sitk_post_process = sitk.GetImageFromArray(fdg_post_np.astype(np.uint8))
    preds_fdg_sitk_post_process.CopyInformation(preds_fdg_sitk)

    return preds_fdg_sitk_post_process
