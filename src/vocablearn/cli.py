"""Command-line interface for vocablearn.

Provides CLI access to vocabulary tracking, learning, and analysis.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from vocablearn.api import VocabLearn


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize a new vocabulary database."""
    db_path = Path(args.database)

    if db_path.exists() and not args.force:
        print(f"Database already exists: {db_path}", file=sys.stderr)
        print("Use --force to overwrite", file=sys.stderr)
        return 1

    if db_path.exists() and args.force:
        db_path.unlink()

    vocab = VocabLearn(db_path)
    print(f"Initialized vocabulary database: {db_path}")
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    """Record tags from a JSON file."""
    # Read tags from JSON file or stdin
    if args.file == "-":
        data = json.load(sys.stdin)
    else:
        with open(args.file) as f:
            data = json.load(f)

    # Support both single image and batch formats
    if "image_id" in data:
        # Single image format
        images = [data]
    elif "images" in data:
        # Batch format
        images = data["images"]
    else:
        print("Invalid JSON format. Expected 'image_id' or 'images' key", file=sys.stderr)
        return 1

    # Dry run mode - just preview
    if getattr(args, 'dry_run', False):
        print("DRY RUN - No changes will be made")
        print("=" * 40)
        total_tags = 0
        for img_data in images:
            image_id = img_data.get("image_id", "unknown")
            tags = img_data.get("tags", [])
            source = img_data.get("source", args.source)
            total_tags += len(tags)
            print(f"\nImage: {image_id}")
            print(f"  Source: {source}")
            print(f"  Tags ({len(tags)}):")
            for tag in tags[:10]:
                label = tag.get("label", "")
                conf = tag.get("confidence", "N/A")
                conf_str = f"{conf:.2f}" if isinstance(conf, float) else conf
                print(f"    - {label} ({conf_str})")
            if len(tags) > 10:
                print(f"    ... and {len(tags) - 10} more")
        print(f"\nTotal: {total_tags} tags from {len(images)} images")
        return 0

    # Normal mode - record to database
    vocab = VocabLearn(args.database)
    total_events = 0
    for img_data in images:
        image_id = img_data["image_id"]
        tags = img_data.get("tags", [])
        source = img_data.get("source", args.source)

        event_ids = vocab.record_tags(image_id, tags, source)
        total_events += len(event_ids)

        if args.verbose:
            print(f"Recorded {len(event_ids)} tags for {image_id}")

    print(f"Recorded {total_events} tag events from {len(images)} images")
    return 0


def cmd_feedback(args: argparse.Namespace) -> int:
    """Record human feedback on a tag."""
    vocab = VocabLearn(args.database)

    vocab.record_feedback(
        image_id=args.image_id,
        tag_label=args.tag,
        correct=args.correct,
        verified_by=args.user,
    )

    status = "correct" if args.correct else "incorrect"
    print(f"Recorded feedback: {args.tag} on {args.image_id} is {status}")
    return 0


def cmd_lookup(args: argparse.Namespace) -> int:
    """Look up a tag in the vocabulary."""
    vocab = VocabLearn(args.database)

    tag = vocab.get_tag(args.tag)
    if tag is None:
        print(f"Tag not found: {args.tag}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(tag.to_dict(), indent=2))
    else:
        print(f"Tag: {tag.label}")
        print(f"  Occurrences: {tag.total_occurrences}")
        print(f"  Prior confidence: {tag.prior_confidence:.3f}")
        print(f"  Confirmed correct: {tag.confirmed_correct}")
        print(f"  Confirmed incorrect: {tag.confirmed_incorrect}")
        print(f"  Sample size: {tag.sample_size}")
        print(f"  Is compound: {tag.is_compound}")
        if tag.ram_plus_hits > 0:
            print(f"  RAM++ detections: {tag.ram_plus_hits}")
        if tag.florence_2_hits > 0:
            print(f"  Florence-2 detections: {tag.florence_2_hits}")
        if tag.siglip_verified > 0:
            print(f"  SigLIP verifications: {tag.siglip_verified}")
        if tag.model_agreement_count > 0:
            print(f"  Model agreement: {tag.model_agreement_count}")

    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Search the vocabulary."""
    vocab = VocabLearn(args.database)

    tags = vocab.search_vocabulary(
        query=args.query,
        min_occurrences=args.min_occurrences,
        min_confidence=args.min_confidence,
        is_compound=args.compound,
        limit=args.limit,
    )

    if args.json:
        print(json.dumps([t.to_dict() for t in tags], indent=2))
    else:
        if not tags:
            print("No tags found matching criteria")
            return 0

        print(f"Found {len(tags)} tags:")
        for tag in tags:
            confidence_str = f"prior={tag.prior_confidence:.2f}"
            compound_str = " [compound]" if tag.is_compound else ""
            print(f"  {tag.label}: {tag.total_occurrences} occurrences, {confidence_str}{compound_str}")

    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Show vocabulary statistics."""
    vocab = VocabLearn(args.database)

    stats = vocab.get_statistics()

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print("Vocabulary Statistics")
        print("=" * 40)
        print(f"Total vocabulary: {stats.get('total_vocabulary', 0)}")
        print(f"Total events: {stats.get('total_events', 0)}")
        print(f"Total feedback: {stats.get('total_feedback', 0)}")
        print(f"Calibrated tags: {stats.get('calibrated_tags', 0)}")
        print(f"Avg prior confidence: {stats.get('avg_prior_confidence', 0.5):.3f}")
        print(f"RAM++ unique tags: {stats.get('ram_plus_unique_tags', 0)}")
        print(f"Florence-2 unique tags: {stats.get('florence_2_unique_tags', 0)}")
        print(f"Model agreement tags: {stats.get('model_agreement_tags', 0)}")

    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Rebuild calibration models."""
    vocab = VocabLearn(args.database)

    # Update priors first
    priors_updated = vocab.update_priors(min_samples=args.min_samples)
    print(f"Updated priors for {priors_updated} tags")

    # Rebuild calibrators
    calibrators_rebuilt = vocab.rebuild_calibrators(
        min_samples=args.min_samples,
        tags=args.tags.split(",") if args.tags else None,
    )
    print(f"Rebuilt {calibrators_rebuilt} calibration models")

    return 0


def cmd_review(args: argparse.Namespace) -> int:
    """Select images for human review."""
    vocab = VocabLearn(args.database)

    candidates = vocab.select_for_review(
        n=args.count,
        strategy=args.strategy,
    )

    if args.json:
        print(json.dumps(candidates, indent=2))
    else:
        if not candidates:
            print("No images need review")
            return 0

        print(f"Selected {len(candidates)} images for review:")
        for i, c in enumerate(candidates, 1):
            print(f"\n{i}. {c['image_id']}")
            print(f"   Priority: {c['priority']}")
            print(f"   Uncertainty: {c['uncertainty_score']:.3f}")
            print(f"   Uncertain tags: {', '.join(c['uncertain_tags'][:5])}")
            if len(c['uncertain_tags']) > 5:
                print(f"   ... and {len(c['uncertain_tags']) - 5} more")

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export vocabulary to JSON."""
    vocab = VocabLearn(args.database)

    vocab.export_vocabulary(args.output)
    print(f"Exported vocabulary to {args.output}")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    """Import vocabulary from JSON."""
    vocab = VocabLearn(args.database)

    imported = vocab.import_vocabulary(
        args.input,
        merge=not args.replace,
    )

    mode = "replaced" if args.replace else "merged"
    print(f"Imported {imported} tags ({mode})")
    return 0


def cmd_export_curves(args: argparse.Namespace) -> int:
    """Export calibration curves to CSV."""
    vocab = VocabLearn(args.database)

    vocab.export_calibration_curves(args.output)
    print(f"Exported calibration curves to {args.output}")
    return 0


def cmd_confidence(args: argparse.Namespace) -> int:
    """Get calibrated confidence for a tag."""
    vocab = VocabLearn(args.database)

    calibrated = vocab.get_calibrated_confidence(
        label=args.tag,
        raw_confidence=args.raw,
        source=args.source,
    )

    if args.json:
        print(json.dumps({
            "tag": args.tag,
            "raw_confidence": args.raw,
            "calibrated_confidence": calibrated,
            "source": args.source,
        }, indent=2))
    else:
        print(f"Tag: {args.tag}")
        print(f"Raw confidence: {args.raw:.3f}")
        print(f"Calibrated confidence: {calibrated:.3f}")
        print(f"Source: {args.source}")

    return 0


def cmd_histogram(args: argparse.Namespace) -> int:
    """Show confidence histogram for calibration analysis."""
    vocab = VocabLearn(args.database)

    # Get calibration data
    from vocablearn.storage.sqlite import SQLiteStorage
    storage = SQLiteStorage(args.database)
    cal_data = storage.get_calibration_data(model=args.model)

    if not cal_data:
        print("No calibration data available", file=sys.stderr)
        return 1

    # Build histogram buckets
    buckets = {i / 10: {"correct": 0, "incorrect": 0} for i in range(10)}

    for point in cal_data:
        bucket = min(9, int(point.raw_confidence * 10)) / 10
        if point.was_correct:
            buckets[bucket]["correct"] += 1
        else:
            buckets[bucket]["incorrect"] += 1

    if args.json:
        output = []
        for bucket, counts in sorted(buckets.items()):
            total = counts["correct"] + counts["incorrect"]
            accuracy = counts["correct"] / total if total > 0 else 0
            output.append({
                "bucket": f"{bucket:.1f}-{bucket + 0.1:.1f}",
                "correct": counts["correct"],
                "incorrect": counts["incorrect"],
                "total": total,
                "accuracy": round(accuracy, 3),
            })
        print(json.dumps(output, indent=2))
    else:
        print(f"Confidence Histogram ({args.model or 'all models'})")
        print("=" * 60)
        print(f"{'Bucket':<12} {'Correct':<10} {'Incorrect':<10} {'Accuracy':<10}")
        print("-" * 60)

        for bucket, counts in sorted(buckets.items()):
            total = counts["correct"] + counts["incorrect"]
            if total == 0:
                continue
            accuracy = counts["correct"] / total
            bar = "#" * int(accuracy * 20)
            print(f"{bucket:.1f}-{bucket + 0.1:.1f}    {counts['correct']:<10} {counts['incorrect']:<10} {accuracy:.1%} {bar}")

    return 0


def cmd_classify(args: argparse.Namespace) -> int:
    """Classify an image using SigLIP zero-shot or tag-based rules."""
    # Check if SigLIP classification is requested
    if hasattr(args, 'siglip') and args.siglip:
        return _classify_siglip(args)

    # Fallback to tag-based classification
    return _classify_tags(args)


def _classify_siglip(args: argparse.Namespace) -> int:
    """Classify using SigLIP zero-shot (better accuracy)."""
    try:
        from vocablearn.ml.classifier import SceneClassifier, SCENE_CATEGORIES
    except ImportError:
        print("SigLIP dependencies not installed. Install with:", file=sys.stderr)
        print("  pip install torch torchvision transformers>=4.47.0", file=sys.stderr)
        return 1

    classifier = SceneClassifier(model_variant=getattr(args, 'model', 'so400m'))

    # Determine category
    category = getattr(args, 'category', 'indoor_outdoor')

    if category == "indoor_outdoor":
        result = classifier.classify_indoor_outdoor(args.image_file)
    elif category == "time_of_day":
        result = classifier.classify_time_of_day(args.image_file)
    elif category == "photo_type":
        result = classifier.classify_photo_type(args.image_file)
    else:
        if category in SCENE_CATEGORIES:
            result = classifier.classify(args.image_file, SCENE_CATEGORIES[category])
        else:
            print(f"Unknown category: {category}", file=sys.stderr)
            return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Classification: {result.label.upper()}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Method: {result.method}")
        print(f"All scores:")
        for label, score in sorted(result.all_scores.items(), key=lambda x: -x[1]):
            print(f"  {label}: {score:.1%}")

    return 0


def _classify_tags(args: argparse.Namespace) -> int:
    """Classify using tag-based rules (fallback)."""
    # Indoor/outdoor classification based on tags
    INDOOR_TAGS = {
        "indoor", "room", "interior", "house", "building", "office",
        "kitchen", "bedroom", "bathroom", "living room", "restaurant",
        "cafe", "bar", "hotel", "lobby", "hallway", "ceiling", "floor",
        "wall", "furniture", "table", "chair", "sofa", "bed", "lamp",
        "window", "door", "curtain", "carpet", "television", "computer",
    }

    OUTDOOR_TAGS = {
        "outdoor", "outside", "sky", "cloud", "sun", "tree", "grass",
        "mountain", "beach", "ocean", "sea", "lake", "river", "forest",
        "park", "garden", "street", "road", "sidewalk", "building exterior",
        "landscape", "nature", "field", "desert", "snow", "rain", "sunset",
        "sunrise", "horizon", "water", "sand", "rock", "hill", "valley",
    }

    # Load tags from JSON file
    with open(args.tags_file) as f:
        data = json.load(f)

    # Extract all tag labels
    all_tags = set()
    results = data.get("results", {})
    for plugin_data in results.values():
        for tag in plugin_data.get("tags", []):
            label = tag.get("label", "").lower()
            if label:
                all_tags.add(label)

    # Score indoor vs outdoor
    indoor_score = len(all_tags & INDOOR_TAGS)
    outdoor_score = len(all_tags & OUTDOOR_TAGS)

    # Determine classification
    if indoor_score > outdoor_score:
        classification = "indoor"
        confidence = indoor_score / (indoor_score + outdoor_score) if (indoor_score + outdoor_score) > 0 else 0.5
    elif outdoor_score > indoor_score:
        classification = "outdoor"
        confidence = outdoor_score / (indoor_score + outdoor_score) if (indoor_score + outdoor_score) > 0 else 0.5
    else:
        classification = "unknown"
        confidence = 0.5

    # Get matched tags
    indoor_matches = list(all_tags & INDOOR_TAGS)
    outdoor_matches = list(all_tags & OUTDOOR_TAGS)

    if args.json:
        print(json.dumps({
            "classification": classification,
            "confidence": round(confidence, 3),
            "method": "tag_rules",
            "indoor_score": indoor_score,
            "outdoor_score": outdoor_score,
            "indoor_matches": indoor_matches,
            "outdoor_matches": outdoor_matches,
        }, indent=2))
    else:
        print(f"Classification: {classification.upper()}")
        print(f"Confidence: {confidence:.1%}")
        print(f"Method: tag_rules")
        print(f"Indoor signals ({indoor_score}): {', '.join(indoor_matches[:5]) or 'none'}")
        print(f"Outdoor signals ({outdoor_score}): {', '.join(outdoor_matches[:5]) or 'none'}")

    return 0


def cmd_cooccurrence(args: argparse.Namespace) -> int:
    """Analyze tag co-occurrence patterns."""
    from vocablearn.ml.cooccurrence import CooccurrenceBooster
    from vocablearn.storage.sqlite import SQLiteStorage

    storage = SQLiteStorage(args.database)
    booster = CooccurrenceBooster(storage)

    # Build matrix
    pairs = booster.build_cooccurrence_matrix(min_count=getattr(args, 'min_count', 2))

    if hasattr(args, 'tag') and args.tag:
        # Show related tags for specific tag
        related = booster.get_related_tags(
            args.tag,
            top_k=getattr(args, 'top_k', 20),
            min_npmi=getattr(args, 'min_npmi', 0.1),
        )

        if args.json:
            print(json.dumps([{"tag": t, "npmi": n} for t, n in related], indent=2))
        else:
            print(f"Tags related to '{args.tag}':")
            print("-" * 40)
            for tag, npmi in related:
                bar = "#" * int(npmi * 20)
                print(f"  {tag:<20} {npmi:.3f} {bar}")
    else:
        # Show top co-occurring pairs
        top_pairs = booster.export_top_pairs(
            top_k=getattr(args, 'top_k', 50),
            min_count=getattr(args, 'min_count', 5),
        )

        if args.json:
            print(json.dumps(top_pairs, indent=2))
        else:
            print(f"Top {len(top_pairs)} co-occurring tag pairs:")
            print("-" * 60)
            print(f"{'Tag A':<20} {'Tag B':<20} {'Count':<8} {'NPMI':<8}")
            print("-" * 60)
            for pair in top_pairs[:30]:
                print(f"{pair['tag_a']:<20} {pair['tag_b']:<20} {pair['joint_count']:<8} {pair['npmi']:.3f}")

    return 0


def cmd_embeddings(args: argparse.Namespace) -> int:
    """Manage image embeddings for similarity/duplicate detection."""
    try:
        from vocablearn.ml.embeddings import EmbeddingIndex
    except ImportError:
        print("SigLIP dependencies not installed. Install with:", file=sys.stderr)
        print("  pip install torch torchvision transformers>=4.47.0", file=sys.stderr)
        return 1

    index = EmbeddingIndex(model_variant=getattr(args, 'model', 'so400m'))

    # Load existing index if available
    index_path = Path(args.database).with_suffix('.embeddings.json')
    if index_path.exists():
        loaded = index.load(index_path)
        print(f"Loaded {loaded} existing embeddings")

    # Add new images if provided
    if hasattr(args, 'add') and args.add:
        for image_path in args.add:
            path = Path(image_path)
            if path.exists():
                image_id = index.add_image(path)
                print(f"Added: {image_id}")
            else:
                print(f"Not found: {image_path}", file=sys.stderr)
        index.save(index_path)
        print(f"Saved index ({index.size} images)")

    # Find duplicates
    if hasattr(args, 'find_duplicates') and args.find_duplicates:
        threshold = getattr(args, 'threshold', 0.95)
        duplicates = index.find_duplicates(threshold=threshold)

        if args.json:
            print(json.dumps([d.to_dict() for d in duplicates], indent=2))
        else:
            if duplicates:
                print(f"Found {len(duplicates)} potential duplicates (threshold={threshold:.0%}):")
                for dup in duplicates[:20]:
                    print(f"  {dup.image_a[:30]}... <-> {dup.image_b[:30]}...")
                    print(f"    Similarity: {dup.similarity:.1%}")
            else:
                print(f"No duplicates found (threshold={threshold:.0%})")

    # Find similar to specific image
    if hasattr(args, 'similar_to') and args.similar_to:
        top_k = getattr(args, 'top_k', 10)
        similar = index.find_similar(args.similar_to, top_k=top_k)

        if args.json:
            print(json.dumps([s.to_dict() for s in similar], indent=2))
        else:
            print(f"Images similar to '{args.similar_to}':")
            for result in similar:
                print(f"  {result.image_id}: {result.score:.1%}")

    return 0


def cmd_duplicates(args: argparse.Namespace) -> int:
    """Find duplicate images based on tag similarity."""
    import hashlib

    vocab = VocabLearn(args.database)
    from vocablearn.storage.sqlite import SQLiteStorage
    storage = SQLiteStorage(args.database)

    # Get all images with their tags
    with storage._connection() as conn:
        cursor = conn.execute("""
            SELECT DISTINCT image_id FROM tag_events
        """)
        image_ids = [row[0] for row in cursor.fetchall()]

    if not image_ids:
        print("No images in database")
        return 0

    # Build tag fingerprint for each image
    fingerprints: dict[str, set[int]] = {}
    for image_id in image_ids:
        events = storage.get_events_for_image(image_id)
        tag_ids = frozenset(e.tag_id for e in events)
        fingerprints[image_id] = tag_ids

    # Find similar images
    duplicates = []
    checked = set()

    for img1, tags1 in fingerprints.items():
        if img1 in checked:
            continue

        for img2, tags2 in fingerprints.items():
            if img1 >= img2 or img2 in checked:
                continue

            # Calculate Jaccard similarity
            intersection = len(tags1 & tags2)
            union = len(tags1 | tags2)
            similarity = intersection / union if union > 0 else 0

            if similarity >= args.threshold:
                duplicates.append({
                    "image1": img1,
                    "image2": img2,
                    "similarity": round(similarity, 3),
                    "shared_tags": intersection,
                })

    if args.json:
        print(json.dumps(duplicates, indent=2))
    else:
        if not duplicates:
            print(f"No duplicates found (threshold: {args.threshold:.0%})")
            return 0

        print(f"Found {len(duplicates)} potential duplicates (threshold: {args.threshold:.0%})")
        print("-" * 60)
        for dup in duplicates[:20]:  # Show top 20
            print(f"  {dup['image1'][:16]}... <-> {dup['image2'][:16]}...")
            print(f"    Similarity: {dup['similarity']:.0%} ({dup['shared_tags']} shared tags)")

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="vocablearn",
        description="Vocabulary tracking and learning for image tagging",
    )
    parser.add_argument(
        "-d", "--database",
        default="vocabulary.db",
        help="Path to vocabulary database (default: vocabulary.db)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init
    p_init = subparsers.add_parser("init", help="Initialize a new vocabulary database")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing database")
    p_init.set_defaults(func=cmd_init)

    # record
    p_record = subparsers.add_parser("record", help="Record tags from JSON")
    p_record.add_argument("file", help="JSON file with tags (or - for stdin)")
    p_record.add_argument("-s", "--source", default="ensemble", help="Source model")
    p_record.add_argument("--dry-run", action="store_true", help="Preview what would be recorded without writing")
    p_record.set_defaults(func=cmd_record)

    # feedback
    p_feedback = subparsers.add_parser("feedback", help="Record human feedback")
    p_feedback.add_argument("image_id", help="Image identifier")
    p_feedback.add_argument("tag", help="Tag label")
    p_feedback.add_argument(
        "correct",
        type=lambda x: x.lower() in ("true", "1", "yes", "correct"),
        help="Whether tag is correct (true/false)",
    )
    p_feedback.add_argument("-u", "--user", help="User identifier")
    p_feedback.set_defaults(func=cmd_feedback)

    # lookup
    p_lookup = subparsers.add_parser("lookup", help="Look up a tag")
    p_lookup.add_argument("tag", help="Tag label to look up")
    p_lookup.add_argument("--json", action="store_true", help="Output as JSON")
    p_lookup.set_defaults(func=cmd_lookup)

    # search
    p_search = subparsers.add_parser("search", help="Search vocabulary")
    p_search.add_argument("-q", "--query", help="Search query (prefix match)")
    p_search.add_argument("--min-occurrences", type=int, default=0, help="Minimum occurrences")
    p_search.add_argument("--min-confidence", type=float, default=0.0, help="Minimum confidence")
    p_search.add_argument("--compound", action="store_true", default=None, help="Only compound tags")
    p_search.add_argument("-l", "--limit", type=int, default=100, help="Maximum results")
    p_search.add_argument("--json", action="store_true", help="Output as JSON")
    p_search.set_defaults(func=cmd_search)

    # stats
    p_stats = subparsers.add_parser("stats", help="Show vocabulary statistics")
    p_stats.add_argument("--json", action="store_true", help="Output as JSON")
    p_stats.set_defaults(func=cmd_stats)

    # calibrate
    p_calibrate = subparsers.add_parser("calibrate", help="Rebuild calibration models")
    p_calibrate.add_argument("--min-samples", type=int, default=50, help="Minimum samples for calibration")
    p_calibrate.add_argument("--tags", help="Specific tags to calibrate (comma-separated)")
    p_calibrate.set_defaults(func=cmd_calibrate)

    # review
    p_review = subparsers.add_parser("review", help="Select images for review")
    p_review.add_argument("-n", "--count", type=int, default=10, help="Number of images")
    p_review.add_argument(
        "-s", "--strategy",
        choices=["uncertainty", "diversity", "high_volume"],
        default="uncertainty",
        help="Selection strategy",
    )
    p_review.add_argument("--json", action="store_true", help="Output as JSON")
    p_review.set_defaults(func=cmd_review)

    # export
    p_export = subparsers.add_parser("export", help="Export vocabulary to JSON")
    p_export.add_argument("output", help="Output file path")
    p_export.set_defaults(func=cmd_export)

    # import
    p_import = subparsers.add_parser("import", help="Import vocabulary from JSON")
    p_import.add_argument("input", help="Input file path")
    p_import.add_argument("--replace", action="store_true", help="Replace instead of merge")
    p_import.set_defaults(func=cmd_import)

    # export-curves
    p_curves = subparsers.add_parser("export-curves", help="Export calibration curves")
    p_curves.add_argument("output", help="Output CSV file path")
    p_curves.set_defaults(func=cmd_export_curves)

    # confidence
    p_conf = subparsers.add_parser("confidence", help="Get calibrated confidence")
    p_conf.add_argument("tag", help="Tag label")
    p_conf.add_argument("raw", type=float, help="Raw confidence score")
    p_conf.add_argument("-s", "--source", default="ram_plus", help="Source model")
    p_conf.add_argument("--json", action="store_true", help="Output as JSON")
    p_conf.set_defaults(func=cmd_confidence)

    # histogram
    p_hist = subparsers.add_parser("histogram", help="Show confidence histogram")
    p_hist.add_argument("-m", "--model", help="Filter by model (ram_plus, florence_2, etc.)")
    p_hist.add_argument("--json", action="store_true", help="Output as JSON")
    p_hist.set_defaults(func=cmd_histogram)

    # classify (tag-based)
    p_classify = subparsers.add_parser("classify", help="Classify image (indoor/outdoor)")
    p_classify_sub = p_classify.add_subparsers(dest="classify_mode")

    # classify tags (from JSON file)
    p_classify_tags = p_classify_sub.add_parser("tags", help="Classify from tag JSON file")
    p_classify_tags.add_argument("tags_file", help="Path to tags JSON file")
    p_classify_tags.add_argument("--json", action="store_true", help="Output as JSON")
    p_classify_tags.set_defaults(func=cmd_classify)

    # classify siglip (zero-shot from image)
    p_classify_siglip = p_classify_sub.add_parser("siglip", help="Classify using SigLIP zero-shot (better accuracy)")
    p_classify_siglip.add_argument("image_file", help="Path to image file")
    p_classify_siglip.add_argument("-c", "--category", default="indoor_outdoor",
                                   choices=["indoor_outdoor", "time_of_day", "photo_type"],
                                   help="Classification category")
    p_classify_siglip.add_argument("-m", "--model", default="so400m", help="SigLIP model variant")
    p_classify_siglip.add_argument("--json", action="store_true", help="Output as JSON")
    p_classify_siglip.set_defaults(func=cmd_classify, siglip=True)

    # duplicates (tag-based)
    p_dup = subparsers.add_parser("duplicates", help="Find duplicate images by tag similarity")
    p_dup.add_argument("-t", "--threshold", type=float, default=0.8, help="Similarity threshold (0-1)")
    p_dup.add_argument("--json", action="store_true", help="Output as JSON")
    p_dup.set_defaults(func=cmd_duplicates)

    # cooccurrence
    p_cooc = subparsers.add_parser("cooccurrence", help="Analyze tag co-occurrence patterns")
    p_cooc.add_argument("-t", "--tag", help="Show tags related to this tag")
    p_cooc.add_argument("-k", "--top-k", type=int, default=20, help="Number of results")
    p_cooc.add_argument("--min-count", type=int, default=2, help="Minimum co-occurrence count")
    p_cooc.add_argument("--min-npmi", type=float, default=0.1, help="Minimum NPMI threshold")
    p_cooc.add_argument("--json", action="store_true", help="Output as JSON")
    p_cooc.set_defaults(func=cmd_cooccurrence)

    # embeddings
    p_emb = subparsers.add_parser("embeddings", help="Manage image embeddings for similarity")
    p_emb.add_argument("--add", nargs="+", help="Add image(s) to index")
    p_emb.add_argument("--find-duplicates", action="store_true", help="Find duplicate images")
    p_emb.add_argument("--similar-to", help="Find images similar to this image")
    p_emb.add_argument("-t", "--threshold", type=float, default=0.95, help="Similarity threshold")
    p_emb.add_argument("-k", "--top-k", type=int, default=10, help="Number of results")
    p_emb.add_argument("-m", "--model", default="so400m", help="SigLIP model variant")
    p_emb.add_argument("--json", action="store_true", help="Output as JSON")
    p_emb.set_defaults(func=cmd_embeddings)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
