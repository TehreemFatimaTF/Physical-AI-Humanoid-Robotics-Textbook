import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import {useBaseUrl} from '@docusaurus/useBaseUrl';
import styles from './ModuleCard.module.css';

/**
 * ModuleCard component for the Physical AI & Humanoid Robotics platform.
 *
 * This component displays a card for a Physical AI module with title, description,
 * and navigation to the module's chapters.
 */
export default function ModuleCard({title, description, to, icon, className}) {
  return (
    <div className={clsx('col col--6 margin-bottom--lg', className)}>
      <Link
        to={to}
        className={clsx('card', styles.moduleCard)}
        style={{ textDecoration: 'none' }}
      >
        <div className="card__header">
          <h3 className={clsx(styles.moduleCardTitle)}>
            {icon && <span className={styles.moduleIcon}>{icon}</span>}
            {title}
          </h3>
        </div>
        <div className="card__body">
          <p>{description}</p>
        </div>
        <div className="card__footer">
          <button className="button button--secondary button--block">
            Explore Modules
          </button>
        </div>
      </Link>
    </div>
  );
}